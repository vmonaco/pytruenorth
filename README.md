## pytruenorth

``pytruenorth`` is a python package to communicate with and deploy networks to [TrueNorth](http://science.sciencemag.org/content/345/6197/668). It was developed at the [2016 Telluride Neuromorphic Cognition Engineering Workshop](http://telluride.iniforum.ch/).

``pytruenorth`` can be used to train a spiking neural network (SNN) or hand-code a population of leaky integrate-and-fire (LIF) neurons as described by the [TrueNorth neuron model](http://www.research.ibm.com/software/IBMResearch/multimedia/IJCNN2013.neuron-model.pdf). The resulting model can be deployed to either an NS1e (teh TrueNorth hardware) or NSCS (the TrueNorth simulator). Spike outputs can be streamed from the NS1e for real-time system deployment.

## Install

Development version:

    $ pip install git+https://github.com/vmonaco/pytruenorth

Dependencies are: Python 3, numpy, and ujson. It is recommended to use [Anaconda](https://www.continuum.io/downloads) and create a virtual env with the above dependencies installed. For example, after Anaconda is installed:

    $ conda create -n tn python=3 anaconda
    
Then,

    $ source activate tn

and install the package using the command above.

## Example: spiking neural network classifier

Train a probabilistic network on the MNIST dataset and deploy to TrueNorth (either NSCS or an NS1e).

```python
# pytruenorth contains a function to load the MNIST dataset
mnist = load_mnist()

# The FrameClassifier5Core class implements a 5 core network
model = FrameClassifier5Core()

# Make a dir for the TF and TN models
basedir = 'MNIST_5core'
if not os.path.exists(basedir):
    os.mkdir(basedir)

# Directory where to store tensorflow checkpoint files
tfdir = os.path.join(basedir, 'tf-model')

# Directory where to store TrueNorth configuration files
tndir = os.path.join(basedir, 'tn-model')

# Train the model
model.train(mnist.train, mnist.validation,
            dirpath=tfdir, num_epochs=1000, checkpoint_steps=10,
            batch_size=300, report_steps=10, deploy_steps=10)

# High-precision probabilistic network performance
model.test(mnist.test, batch_size=100)

# Low-precision spiking network performance, deploys to NSCS by default
model.deploy(mnist.test, dirpath=tndir)
```

## Example: tonic spiking neurons

Create and configure TrueNorth chip with tonic spiking neurons across two cores.

```python
from pytruenorth import TrueNorthChip, NEURON_DEST_OUT

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
```

Optionally define a step function to process spikes on each time step. The step function should return False if execution should halt (before the number of time steps is reached) and True otherwise.

```python
# Step function generator that simulates a long-running function and 
# halts execution when a threshold is reached
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
```

To run a TrueNorth chip on nscs, the path of the correct openmpi libs required by nscs must be on your LD_LIBRARY_PATH. pytruenorth tries a few default locations, but may not be able to locate them. nscs must also be visible on your PATH variable.

```python
# Run on NSCS for fixed number of time steps
spikes_out_nscs1 = chip.run_nscs(T=100, spikes_in=spikes_in)

# Use the step function to terminate
spikes_out_nscs2 = chip.run_nscs(T=10000, step_fn=step_fn_closure(), spikes_in=spikes_in)
```

The same chip configuration can be deployed to an NS1e.

```python
# Run on TrueNorth for a fixed number of time steps
spikes_out_tn1 = chip.run_tn(T=100, tnhost='truenorth', spikes_in=spikes_in)

# Use the step function to terminate
spikes_out_tn2 = chip.run_tn(T=10000, tnhost='truenorth', spikes_in=spikes_in,
                             step_fn=step_fn_closure())
```

If the NS1e deployment doesn't work, you may have to specify the spike destination host and port (from the perspective of the NS1e) through the udp_spike_destination_host and udp_spike_destination_port parameters. Also be sure that the NS1e can be accessed via an ssh public key: the tnuser identity file must visible to ssh.

Summarize the output spikes. We expect:

* Neuron (0,0): period 10, phase 0
* Neuron (0,1): period 10, phase 9
* Neuron (0,2): period 6, phase 0
* Neuron (1,7): period 5, phase 1 with spikes on axon 7 every tick

```python
def spike_summary(name, spikes):
    print('%s spike count: %d' % (name, len(spikes)))
    for dest_axon in [0, 1, 2, 123]:
        period, phase = extract_period_phase(spikes, NEURON_DEST_OUT, dest_axon)
        print('Neuron %d, Axon %d: Period %.1f, Phase %.1f' % (NEURON_DEST_OUT, dest_axon, period, phase))

spike_summary('NSCS1', spikes_out_nscs1)
spike_summary('NSCS2', spikes_out_nscs2)
spike_summary('TN1', spikes_out_tn1)
spike_summary('TN2', spikes_out_tn2)
```

See examples/ for more examples.
