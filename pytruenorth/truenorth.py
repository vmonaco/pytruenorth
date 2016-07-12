import os
import sys
import time
import queue
import shutil
import socket
import struct
import datetime
import threading
import subprocess
import numpy as np
import ujson as json
import multiprocessing as mp

from .util import call_or, maybe_mkdir

TN_MODEL_JSON_NAME = 'model.json'
TN_MODEL_TNBM_NAME = 'model.tnbm'

TN_CONFIG_NAME = 'config.json'
SPIKES_IN_NAME = 'spikes_in.sfri'
SPIKES_OUT_NAME = 'spikes_out0.sfri'

MAX_NUM_SPIKES_PER_BUFFER = 2040
MAX_NUM_SPIKES_PER_RECEIVE_BUFFER = 4096
TNSF_BUFFER_HEADER_SIZE = 4
SPIKE_TIME_LSB = 2
BUFFER_QUEUE_SIZE = 1000

NEURON_CLASS = 'NeuronGeneral'

SPIKES_HEADER = \
    """{"class":"SpikeWriter","encoding":"TEXT","nscsVersion":"1.4.0"}
    {"version":"sfto1.0", "defaultDestinationDelay" : 1}"""

NUM_NEURONS = 256
NUM_AXONS = 256
NUM_AXON_TYPES = 4

RESET_MODE_NORMAL = 0
RESET_MODE_LINEAR = 1
RESET_MODE_NONE = 2

PRNG_MIN, PRNG_MAX = np.iinfo(np.uint8).min, np.iinfo(np.uint8).max

NEURON_DEST_OUT = -32  # Neuron spike output destination magic number

NEURON_PARAM_NAMES = (
    'sigma0',
    'sigma1',
    'sigma2',
    'sigma3',
    's0',
    's1',
    's2',
    's3',
    'b0',
    'b1',
    'b2',
    'b3',
    'sigma_lambda',
    'lambda',
    'c_lambda',
    'epsilon',
    'alpha',
    'beta',
    'TM',
    'gamma',
    'kappa',
    'sigma_VR',
    'VR',
    'V',
)

DEFAULT_CONFIG = {
    'monitorStack': 'MonitorTnk',
    'MonitorTnk.startingCoreId': -32,
    'MonitorTnk.outputFileName': 'spikes_out',
    'MonitorTnk.outputFileEncoding': 'RAWO6',
    'MonitorTnk.outputFileExtension': 'sfri',

    'tickPeriod': '1',  # ms, must be string
    'tickCount': 0,
    'randSeedVal': 1,
    'inputFileName': '',
    'modelFileName': '',

    'stop_on_spike_mismatch': True,
    'print_output_spikes_mode': 0,
    'discardDisconnectedDendrites': False,

    'hideMessages': False,
    'showProgressDots': False,
    'showCoreState': False,
    'debug_mode': False,

    'networkClass': 'NetworkBus',
    'udp_spike_destination_port': 0,
    'udp_spike_destination_host': '',
    # 'tcpDataInputPort': 0,
    # 'tcpSpikeDestinationHost': 'loopback',
}

# Add default nscs install locations to PATH
os.environ['PATH'] = ':'.join([
    os.environ['HOME'] + '/truenorth/usr/local/bin/',
    '/usr/local/nscs/bin/',
    os.getenv('PATH', '')
])

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()


class TrueNorthCore(object):
    """Represents a single core on a TrueNorth chip"""

    def __init__(self, id, num_neurons=NUM_NEURONS, num_axons=NUM_AXONS, num_axon_types=NUM_AXON_TYPES):
        self.id = id

        self.seed = 1

        # Number of neurons on this core
        self.num_neurons = num_neurons
        self.num_axons = num_axons
        self.num_axon_types = num_axon_types

        self.sigma = np.ones(shape=(num_axons, num_axon_types), dtype=np.int8)
        self.s = np.zeros(shape=(num_axons, num_axon_types), dtype=np.uint8)
        self.b = np.zeros(shape=(num_axons, num_axon_types), dtype=np.bool)

        self.sigma_lambda = - np.ones(num_neurons, dtype=np.int8)
        self.lambda_ = np.zeros(num_neurons, dtype=np.uint8)
        self.c_lambda = np.zeros(num_neurons, dtype=np.bool)

        self.epsilon = np.zeros(num_neurons, dtype=np.bool)

        self.alpha = np.zeros(num_neurons, dtype=np.uint32)
        self.beta = np.zeros(num_neurons, dtype=np.uint32)

        self.TM = np.zeros(num_neurons, dtype=np.uint8)

        self.gamma = np.zeros(num_neurons, dtype=np.uint8)
        self.kappa = np.zeros(num_neurons, dtype=np.bool)

        self.sigma_VR = np.ones(num_neurons, dtype=np.int8)
        self.VR = np.zeros(num_neurons, dtype=np.uint8)
        self.V = np.zeros(num_neurons, dtype=np.int32)

        # Synaptic crossbar
        self.w = np.zeros(shape=(num_axons, num_neurons), dtype=np.bool)

        self.axon_type = np.zeros(num_axons, dtype=np.uint8)

        # Neuron connections
        self.dendrite = np.arange(num_neurons, dtype=np.int)
        self.dest_core = - np.ones(num_neurons, dtype=np.int)  # Max 4096
        self.dest_axon = - np.ones(num_neurons, dtype=np.int)  # Max 256
        self.dest_delay = np.ones(num_neurons, dtype=np.uint8)  # Max 15

    def neuron_params(self, j):
        """Return a tuple of neuron j's params, same order as NEURON_PARAM_NAMES"""

        return (
            int(self.sigma[j, 0]),
            int(self.sigma[j, 1]),
            int(self.sigma[j, 2]),
            int(self.sigma[j, 3]),

            int(self.s[j, 0]),
            int(self.s[j, 1]),
            int(self.s[j, 2]),
            int(self.s[j, 3]),

            bool(self.b[j, 0]),
            bool(self.b[j, 1]),
            bool(self.b[j, 2]),
            bool(self.b[j, 3]),

            int(self.sigma_lambda[j]),
            int(self.lambda_[j]),
            bool(self.c_lambda[j]),

            bool(self.epsilon[j]),

            int(self.alpha[j]),
            int(self.beta[j]),

            int(self.TM[j]),

            int(self.gamma[j]),
            bool(self.kappa[j]),

            int(self.sigma_VR[j]),
            int(self.VR[j]),
            int(self.V[j]),
        )

    def neuron_types(self):
        """Returns a dict of neuron types, keyed by a unique name"""

        neuron_dict = {}  # Neuron configuration
        neuron_list = []  # Connects neuron types to destination axons
        for j in range(self.num_neurons):
            neuron_params = self.neuron_params(j)
            neuron_name = str(hash(neuron_params))
            if neuron_name not in neuron_dict.keys():
                neuron_dict[neuron_name] = dict(zip(NEURON_PARAM_NAMES, self.neuron_params(j)))
                neuron_dict[neuron_name]['name'] = neuron_name

            neuron_list.append({
                'type': neuron_name,
                'dendrite': int(self.dendrite[j]),
                'destCore': int(self.dest_core[j]),
                'destAxon': int(self.dest_axon[j]),
                'delay': int(self.dest_delay[j])
            })

        return neuron_dict, neuron_list

    def crossbar_rows(self):

        """Returns a list of crossbar rows"""
        rows = []
        for i in range(self.num_neurons):
            rows.append({
                'type': 'S%d' % self.axon_type[i],
                'synapses': (' '.join(map(lambda x: format(x, '02x'), np.packbits(self.w[i, :])))).upper()
            })

        return rows

    def to_dict(self):
        """Return the core configuration dict and """

        neuron_dict, neuron_list = self.neuron_types()

        core = {
            'id': self.id,
            'rngSeed': int(self.seed),
            'timeScaleExponent': 0,
            'neurons': neuron_list,
            'crossbar': {'rows': self.crossbar_rows()}
        }

        return core, neuron_dict

# TODO: spike streaming. This function basically works, but synchronization with the chip needs to be addressed.
# def encode_tnsf_buffer(t, spikes, T, buffer_id_obj):
#     """
#     t is the current time
#     spikes is a list of (time, core, axon) spikes to send
#     T is the number of ticks to advance (currently either 0 or 1)
#     """
#     buffer_size = TNSF_BUFFER_HEADER_SIZE + len(spikes)
#     buffer = [0] * buffer_size
#
#     spike_buffer_version = 3  # 8-bits. Version 3 is a TNSF ready to DMA
#     port_id = 0  # 4-bits: Injection port
#     buffer_id = buffer_id_obj.id  # 6-bits: Incremented once for every buffer sent to detect lost buffers
#     spike_count = len(spikes)  # : 14-bits (Currently supports up to 4096 spikes per buffer)
#     header_word_0 = \
#         ((spike_count & 0x3fff) << 18) | \
#         ((buffer_id & 0x3f) << 12) | \
#         ((port_id & 0xf) << 8) | \
#         ((spike_buffer_version & 0xff) << 0)
#
#     buffer[0] = header_word_0
#     buffer[1] = t
#     buffer[2] = T
#     buffer[3] = 0
#
#     for i, spike in zip(range(4, buffer_size), spikes):
#         core_x = spike[1] // 64
#         core_y = spike[1] % 64
#         axon = spike[2]
#
#         buffer[i] = 0 | \
#                     (core_x & 0x1FF) << 23 | \
#                     (core_y & 0x1FF) << 14 | \
#                     (axon & 0xFF) << 6 | \
#                     ((t + 1) % 16) << SPIKE_TIME_LSB
#
#     buffer_id_obj.id += 1
#     return struct.pack('>I' * buffer_size, *buffer)


def decode_tnsf_buffer(buffer):
    """
    buffer is a bytearray from TrueNorth
    """
    # Unpack the header
    buffer_header = struct.unpack('IIII', buffer[:TNSF_BUFFER_HEADER_SIZE * 4])

    # spike_buffer_version = (buffer_header[0] >> 0) & 0xff
    # tn_port = (buffer_header[0] >> 8) & 0xf
    # buffer_id = (buffer_header[0] >> 12) & 0x3f
    spike_count = (buffer_header[0] >> 18) & 0x3fff
    tick_observed = buffer_header[1]
    offset_x = (buffer_header[2] >> 16) & 0xffff
    offset_y = (buffer_header[2] >> 0) & 0xffff
    # CORE_OUTPUT_MAP_HEIGHT = (buffer_header[3] >> 16) & 0xffff
    CPE_CORE_OUTPUT_OFFSET = (buffer_header[3] >> 0) & 0xffff

    spikes_words = struct.unpack('I' * spike_count, buffer[TNSF_BUFFER_HEADER_SIZE * 4:])

    spikes = np.zeros(shape=(spike_count, 3), dtype=np.int32)
    for i, spike_word in enumerate(spikes_words):
        # debug = (spike_word >> 1) & 0x1
        # t = (spike_word >> 2) & 0xF
        axon = (spike_word >> 6) & 0xFF

        # core_y = (((spike_word << 9)) >> (9 + 14)) + offset_y
        # core_x = (spike_word >> 23) + offset_x
        # dest_core_id = (core_x + 1) * CORE_OUTPUT_MAP_HEIGHT - core_y + CPE_CORE_OUTPUT_OFFSET
        # delivery_time = tick_observed + ((t - tick_observed) & 0xf)

        core_y = ((int(spike_word << 9)) >> (9 + 14)) + offset_y
        core_x = (spike_word >> 23) + offset_x
        dest_core = CPE_CORE_OUTPUT_OFFSET - (core_x + core_y + 1)

        spikes[i] = tick_observed, dest_core, axon

    return spikes


class TrueNorthChip(object):
    def __init__(self, num_cores, num_neurons=NUM_NEURONS):

        # Number of cores on the chip
        self.num_cores = num_cores

        # Number of neurons on each core
        self.neuron_class = NEURON_CLASS
        self.num_neurons = num_neurons
        self.cores = [TrueNorthCore(i, num_neurons) for i in range(num_cores)]

        # Time
        self.t = np.uint16(0)

    def run_tn(self, T, tnhost, udp_spike_destination_host=None, udp_spike_destination_port=5000,
               step_fn=lambda spikes: True,
               spikes_in=[], return_spikes_out=True, cleanup_after=True, root_dir='/tmp/pytruenorth',
               remote_tnk_ctrl='/usr/local/share/truenorth/tnk/tnk_ctrl', config_args={}, verbose=False):

        if step_fn is None:
            step_fn = lambda spikes_out_t: True

        # Try to detect the destination host (probably this machine) automagically
        if udp_spike_destination_host is None:
            udp_spike_destination_host = socket.gethostbyname(socket.gethostname())

        # Use the same dir on local and remote
        config_dir = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H-%M-%S')

        # Same local and remote path
        local_path = remote_path = os.path.join(root_dir, config_dir)
        config_fname = os.path.join(local_path, TN_CONFIG_NAME)
        model_json_fname = os.path.join(local_path, TN_MODEL_JSON_NAME)
        model_tnbm_fname = os.path.join(local_path, TN_MODEL_TNBM_NAME)
        spikes_in_fname = os.path.join(local_path, SPIKES_IN_NAME)

        # Create the dirs
        call_or("mkdir -p %s" % local_path, error=True)
        call_or("ssh %s 'mkdir -p %s'" % (tnhost, remote_path), error=True)

        # Create the config
        config = DEFAULT_CONFIG.copy()
        config['tickCount'] = T
        config['udp_spike_destination_host'] = udp_spike_destination_host
        config['udp_spike_destination_port'] = udp_spike_destination_port

        # Create the model file
        self.model2json(model_json_fname)

        # Try to use a binary model for faster load time
        env = dict(os.environ)
        env['OMP_NUM_THREADS'] = str(min(mp.cpu_count(), self.num_cores))
        p = subprocess.call(
            'json2tnk %s %s' % (os.path.join(local_path, model_json_fname), os.path.join(local_path, model_tnbm_fname)),
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        if p == 0:
            config['modelFileName'] = TN_MODEL_TNBM_NAME
        else:
            # Fallback to json model file
            print('Warning: unable to convert model json2tnk')
            config['modelFileName'] = TN_MODEL_JSON_NAME

        # Dump the input spikes
        self.spikes2raw(spikes_in, spikes_in_fname)
        config['inputFileName'] = SPIKES_IN_NAME

        # Generate the config file
        config.update(config_args)
        json.dump(config, open(config_fname, 'wt'), escape_forward_slashes=False, indent=4)

        # Copy everything to TrueNorth
        call_or("scp %s/* %s:%s" % (local_path, tnhost, remote_path), error=True)

        # Kill any tnk_ctrl currently running
        call_or("ssh %s 'pkill tnk_ctrl'" % tnhost)

        # Listen for UDP on the port
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(('', udp_spike_destination_port))

        # Start another thread for the callback function
        spikes_out = []
        spike_queue = queue.Queue(BUFFER_QUEUE_SIZE)

        def enqueue_spikes():
            while True:
                try:
                    buffer = sock.recv(MAX_NUM_SPIKES_PER_RECEIVE_BUFFER + TNSF_BUFFER_HEADER_SIZE)
                    if buffer:
                        spike_queue.put(buffer)
                except:
                    break

        # Listen for spikes before starting tnk_ctrl
        producer_thread = threading.Thread(target=enqueue_spikes, daemon=True)
        producer_thread.start()

        # Start tnk_ctrl async and wait until it has initialized by looking for "Waiting for client connection."
        out = sys.stdout if verbose else subprocess.DEVNULL
        tnk_ctrl_proc = subprocess.Popen(
            "ssh %s 'cd %s ; %s +config=%s'" % (tnhost, remote_path, remote_tnk_ctrl, TN_CONFIG_NAME),
            shell=True,
            stdout=out, stderr=out
        )

        print('Ready')
        sys.stdout.flush()

        while True:
            if spike_queue.empty():
                # Check the producer thread
                if not producer_thread.isAlive() or tnk_ctrl_proc.poll() is not None:
                    break
                # Wait for the queue to fill up
                time.sleep(0.001)
            else:
                spikes = []
                while not spike_queue.empty():
                    spikes.append(decode_tnsf_buffer(spike_queue.get()))

                spikes = np.concatenate(spikes)

                if return_spikes_out:
                    spikes_out.append(spikes)

                if not step_fn(spikes):
                    break

        # Main loop ended, kill the subprocess
        tnk_ctrl_proc.kill()

        # Cleanup
        sock.close()
        if cleanup_after:
            call_or("rm -rf %s" % local_path, complain=True)
            call_or("ssh %s 'rm -rf %s'" % (tnhost, remote_path), complain=True)

        if return_spikes_out:
            return np.array([], dtype=np.int32) if len(spikes_out) == 0 else np.concatenate(spikes_out)
        else:
            return

    def run_nscs(self, T, step_fn=lambda spikes: True, spikes_in=[], return_spikes_out=True, cleanup_after=True,
                 dirpath=None, config_args={}, verbose=False):
        """Run NSCS and gather output spikes using the current chip configuration

        """
        # Dir and filenames
        deploy_dir = maybe_mkdir(dirpath)
        config_fname = os.path.join(deploy_dir, TN_CONFIG_NAME)
        model_json_fname = os.path.join(deploy_dir, TN_MODEL_JSON_NAME)
        spikes_in_fname = os.path.join(deploy_dir, SPIKES_IN_NAME)
        spikes_out_fname = os.path.join(deploy_dir, SPIKES_OUT_NAME)

        # Start with the default config
        config = DEFAULT_CONFIG.copy()
        config['tickCount'] = T

        # Create the model file
        self.model2json(model_json_fname)

        # if cores < proc threads, then OMP_NUM_THREADS = num cores
        env = dict(os.environ)
        env['OMP_NUM_THREADS'] = str(min(mp.cpu_count(), self.num_cores))

        # Model json only for nscs
        config['modelFileName'] = TN_MODEL_JSON_NAME

        # Dump the input spikes
        self.spikes2raw(spikes_in, spikes_in_fname)
        config['inputFileName'] = SPIKES_IN_NAME

        # Generate the config file
        config.update(config_args)
        json.dump(config, open(config_fname, 'wt'), escape_forward_slashes=False, indent=4)

        # Touch the output file, read by the producer thread below
        call_or('touch %s' % spikes_out_fname, error=True)

        # Output spikes are written to the file and enqueued by the producer
        spikes_out = []
        spike_queue = queue.Queue(BUFFER_QUEUE_SIZE)

        spike_size = 24

        # Read spikes from the output file
        def enqueue_spikes():
            with open(spikes_out_fname, 'rb') as f:
                # Skip past the header, terminated by 0xFF
                while b'\xFF' != f.read(1):
                    pass

                while True:
                    where = f.tell()
                    spike_buffer = f.read()

                    if spike_buffer == b'':
                        # Nothing new, wait one ms and go back to marker
                        time.sleep(0.001)
                        f.seek(where)
                    else:
                        # Spikes are 3 words (12 bytes)
                        num_spikes = len(spike_buffer) // spike_size

                        # Align the reader with 3-word spike format
                        if (len(spike_buffer) - num_spikes * spike_size) > 0:
                            f.seek(f.tell() - (len(spike_buffer) - num_spikes * spike_size))

                        # Only enqueue full spikes
                        if num_spikes > 0:
                            spike_queue.put(spike_buffer[:spike_size * num_spikes])

        # Listen for spikes before starting tnk_ctrl
        producer_thread = threading.Thread(target=enqueue_spikes, daemon=True)
        producer_thread.start()

        # Run NSCS
        out = sys.stdout if verbose else subprocess.DEVNULL
        nscs_proc = subprocess.Popen(
            'nscs %s' % config_fname,
            cwd=deploy_dir, env=env, shell=True,
            stdout=out, stderr=out
        )

        while True:
            if spike_queue.empty():
                # Check the producer thread
                if not producer_thread.isAlive() or nscs_proc.poll() is not None:
                    break
                # Wait for the queue to fill up
                time.sleep(0.001)
            else:
                spikes = []
                while not spike_queue.empty():
                    spike_buffer = spike_queue.get()
                    num_spikes = len(spike_buffer) // spike_size

                    # RAW06 format, columns [0, 3, 4] are [tick, dest_core, dest_axon]
                    buffer_spikes = np.array(struct.unpack('>' + 'IIIIII' * num_spikes, spike_buffer),
                                             dtype=np.int32).reshape(-1, 6)[:, [0, 3, 4]]
                    spikes.append(buffer_spikes)

                spikes = np.concatenate(spikes)

                if return_spikes_out:
                    spikes_out.append(spikes)

                if not step_fn(spikes):
                    break

        # Main loop ends, kill the subprocess
        nscs_proc.kill()

        if cleanup_after:
            shutil.rmtree(deploy_dir)

        if return_spikes_out:
            return np.array([], dtype=np.int32) if len(spikes_out) == 0 else np.concatenate(spikes_out)
        else:
            return

    def model2json(self, fname=None):
        """Dump the chip configuration to a json string or file

        Reason for taking the slice 1:-1 of the json strings generated below: this is to remove the top level
        brackets and concatenate the model, neuronTypes, and core keys in the same namespace. This isn't really
        json since TN expects a certain order of key/value pairs (must be model, neuronTypes, core), and key names
        are not unique.
        """
        # Add the cores to the chip and maintain a set of neuron types
        neuron_types = {}  # Keyed by neuron name
        core_dicts = []
        for i, core in enumerate(self.cores):
            core_dict, core_neuron_types = core.to_dict()
            core_dicts.append(core_dict)
            neuron_types.update(core_neuron_types)

        model_json = json.dumps({
            'model': {
                'coreCount': self.num_cores,
                'neuronClass': self.neuron_class,
                'crossbarSize': self.num_neurons,

                'crossbarclass': 'CrossbarBinary',
                'networking': 'INTRA',
            },
        }, indent=4)[1:-1]

        neuron_types_json = json.dumps({'neuronTypes': list(neuron_types.values())}, indent=4)[1:-1]
        cores_json = [json.dumps({'core': core_dict}, indent=4)[1:-1] for core_dict in core_dicts]

        json_str = '{\n%s\n}' % ',\n'.join([model_json, neuron_types_json] + cores_json)

        if fname is not None:
            with open(fname, 'w') as f:
                f.write(json_str)

        return json_str

    def spikes2json(self, spikes, fname):
        with open(fname, 'wt') as f:
            f.write(SPIKES_HEADER + '\n')

            for t, dest_core, dest_axon in spikes:
                f.write(
                    """{"spike":{"srcTime":%d,"destCore":%d,"destAxon":%d}}\n""" % (
                        t, dest_core, dest_axon
                    ))
        return

    def spikes2raw(self, spikes, fname):
        with open(fname, 'wb') as f:
            f.write(
                '{"class":"SpikeWriter","nscsVersion":"1.4.0","encoding":"RAWI3","defaultDestinationDelay":1}'.encode(
                    'utf-8'))
            f.write(b'\xff')

            for t, dest_core, dest_axon in spikes:
                f.write(struct.pack('>III', t, dest_core, dest_axon))
        return

    def raw2spikes(self, fname):
        spikes_out = []
        with open(fname, 'rb') as f:
            # Skip to the marker
            while b'\xff' != f.read(1):
                pass

            while True:
                spike = f.read(4 * 3)
                if spike == b'':
                    break
                spikes_out.append(struct.unpack('>III', spike))

        return np.r_[spikes_out]
