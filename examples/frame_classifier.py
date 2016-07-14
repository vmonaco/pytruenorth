import sys
import numpy as np


def run(tnhost, udp_spike_destination_host, udp_spike_destination_port):

    cl = FrameClassifier()
    cl.fit()
    cl.predict()
    cl.deploy()
    return


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    if len(sys.argv) < 4:
        print('Usage: $ python frame_classifier.py [tnhost] [spike_dest_host] [spike_dest_port]')

    tnhost, udp_spike_destination_host, udp_spike_destination_port = sys.argv[1:4]
    run(tnhost, udp_spike_destination_host, int(udp_spike_destination_port))
