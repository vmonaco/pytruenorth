import os
import sys
import time
import numpy as np

from pytruenorth import TrueNorthChip, NEURON_DEST_OUT

# Important: make sure openmpi libs are on your LD_LIBRARY_PATH. For example, check some standard locations.
os.environ['LD_LIBRARY_PATH'] = ':'.join([
    os.getenv('LD_LIBRARY_PATH', ''),
    '/usr/lib64/compat-openmpi/lib/',
    '/usr/lib/openmpi/lib/',
    os.environ['HOME'] + '/lib/openmpi/',
])

# Also important: make sure nscs is on your PATH. This should work if nscs was installed normally.
os.environ['PATH'] = ':'.join([
    os.getenv('PATH', ''),
    os.environ['HOME'] + '/truenorth/usr/local/bin/',
    '/usr/local/nscs/bin/',
])


def extract_period_phase(spikes, dest_core, dest_axon):
    spike_times = spikes[np.where((spikes[:, 1] == dest_core) & (spikes[:, 2] == dest_axon))[0], 0]

    period = np.diff(spike_times[1:]).mean()
    phase = (spike_times[0] + 1) % period

    return period, phase


def run():
    """Tonic spiking neurons across two cores.
    Neuron (0,0): period 10, phase 0
    Neuron (0,1): period 10, phase 9
    Neuron (0,2): period 6, phase 0
    Neuron (1,7): period 5, phase 1 with spikes on axon 7 every tick
    """
    chip = TrueNorthChip(num_cores=2)

    # Neurons 0-2 on core 0 are tonic spiking
    chip.cores[0].alpha[0] = 10
    chip.cores[0].lambda_[0] = 1
    chip.cores[0].sigma_lambda[0] = 1
    chip.cores[0].dest_core[0] = NEURON_DEST_OUT
    chip.cores[0].dest_axon[0] = 0

    chip.cores[0].alpha[1] = 10
    chip.cores[0].V[1] = 1
    chip.cores[0].lambda_[1] = 1
    chip.cores[0].sigma_lambda[1] = 1
    chip.cores[0].dest_core[1] = NEURON_DEST_OUT
    chip.cores[0].dest_axon[1] = 1

    chip.cores[0].alpha[2] = 11
    chip.cores[0].lambda_[2] = 2
    chip.cores[0].sigma_lambda[2] = 1
    chip.cores[0].dest_core[2] = NEURON_DEST_OUT
    chip.cores[0].dest_axon[2] = 2

    # Neuron 7 on core 1 is connected to axon 3
    chip.cores[1].s[:] = [3, 0, 0, 0]
    chip.cores[1].w[3, 7] = 1
    chip.cores[1].alpha[7] = 10
    chip.cores[1].lambda_[7] = 1
    chip.cores[1].sigma_lambda[7] = -1
    chip.cores[1].dest_core[7] = NEURON_DEST_OUT
    chip.cores[1].dest_axon[7] = 123

    # Send spikes to axon 3 on core 1. Spikes are (time, core, axon) tuples
    spikes_in = np.concatenate([[(t, 1, 3), (t, 1, 4)] for t in range(100)])

    # Step function generator that simulates a long-running function and halts execution when a threshold is reached
    def step_fn_closure(num_spikes=55, time_sleep=0.1):
        counter = 0

        def step_fn(spikes):
            nonlocal counter
            counter += len(spikes)

            # Simulate a long-running function
            time.sleep(time_sleep)

            sys.stdout.write('Step function spikes received: %d\n' % len(spikes))
            sys.stdout.flush()

            if counter >= num_spikes:
                return False

            return True

        return step_fn

    # Run on NSCS for fixed number of time steps
    spikes_out_nscs1 = chip.run_nscs(T=100, spikes_in=spikes_in)

    # Use the step function to terminate
    spikes_out_nscs2 = chip.run_nscs(T=10000, step_fn=step_fn_closure(), spikes_in=spikes_in)

    # Run on TrueNorth for a fixed number of time steps
    spikes_out_tn1 = chip.run_tn(T=100, tnhost='tnfob', spikes_in=spikes_in)

    # Use the step function to terminate
    spikes_out_tn2 = chip.run_tn(T=10000, tnhost='tnfob', spikes_in=spikes_in,
                                 step_fn=step_fn_closure())

    def spike_summary(name, spikes):
        print('%s spike count: %d' % (name, len(spikes)))
        for dest_axon in [0, 1, 2, 123]:
            period, phase = extract_period_phase(spikes, NEURON_DEST_OUT, dest_axon)
            print('Neuron %d, Axon %d: Period %.1f, Phase %.1f' % (NEURON_DEST_OUT, dest_axon, period, phase))

    spike_summary('NSCS1', spikes_out_nscs1)
    spike_summary('NSCS2', spikes_out_nscs2)
    spike_summary('TN1', spikes_out_tn1)
    spike_summary('TN2', spikes_out_tn2)

    return


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    run()
