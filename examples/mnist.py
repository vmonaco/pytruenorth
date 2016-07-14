import os
import numpy as np
import tensorflow as tf

from pytruenorth.data import load_mnist
from pytruenorth.trueshadow import FrameClassifier5Core

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(ROOT_DIR, 'models')


def mnist_results(num_tn_reps=5, seed=1234, restore=False):
    """Train and deploy a network using pytruenorth"""
    np.random.seed(seed)

    # mnist is a DataSets object with .train, .test, and .validation members
    mnist = load_mnist()

    # Reset the graph before creating a model
    tf.reset_default_graph()

    model = FrameClassifier5Core()

    # Make a dir for the TF and TN models
    basedir = os.path.join(MODELS_DIR, 'MNIST_5core')
    if not os.path.exists(basedir):
        os.mkdir(basedir)

    # Directory where to store tensorflow checkpoint files
    tfdir = os.path.join(basedir, 'tf-model')

    # Directory where to store TrueNorth configuration files
    tndir = os.path.join(basedir, 'tn-model')

    # Optionally restore a model
    if restore:
        model.load(tfdir)
    else:
        # Train the model
        model.train(mnist.train, mnist.validation,
                    dirpath=tfdir, num_epochs=1000, checkpoint_steps=10,
                    batch_size=300, report_steps=10, deploy_steps=10)

    # Shadow network performance
    shadow_acc = model.test(mnist.test, batch_size=100)

    # Save the results
    shadow_summary = '%04f' % shadow_acc
    print('Shadow ACC:', shadow_summary)
    with open(os.path.join(basedir, 'shadow_acc.txt'), 'wt') as f:
        f.write(shadow_summary)

    # TN network performance for several runs (ensembles are currently not implemented)
    deploy_acc = []
    for _ in range(num_tn_reps):
        deploy_acc.append(model.deploy(mnist.test, dirpath=tndir))

    deploy_summary = '%04f +/- %04f' % (np.mean(deploy_acc), np.std(deploy_acc))
    print('Deployment ACC:', deploy_summary)
    with open(os.path.join(basedir, 'deploy_acc.txt'), 'wt') as f:
        f.write(deploy_summary)

    return


# If you're using tensorflow GPU, make sure the cuda libs (libcudart and libcudnn) are visible on LD_LIBRARY_PATH
if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    mnist_results()
