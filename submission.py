import argparse
from HMM import run_hmm
from alarm import print_alarm_cpd, alarm_model
from carnet import print_carnet_cpd, car_model


def main():
    print("====================== Part 2 - HMM ======================")
    print("---- Run the following commands in the terminal ----")
    print("python3 hmm.py --generate 20")
    print("python3 hmm.py --forward ambiguous_sents.obs")
    print("python3 hmm.py --viterbi ambiguous_sents.obs")

    parser = argparse.ArgumentParser(description='HMM Arguments')

    #  python3 hmm.py --generate 20
    parser.add_argument('--generate', type=int)

    #  python3 hmm.py --forward ambiguous_sents.obs
    parser.add_argument('--forward', type=str)

    #  python3 hmm.py --viterbi ambiguous_sents.obs
    parser.add_argument('--viterbi', type=str)

    args = parser.parse_args()

    run_hmm(args)

    print("============ Part 3 - Bayesian Network (alarm) ===========")
    print_alarm_cpd(alarm_model)

    print("=========== Part 3 - Bayesian Network (carnet) ============")
    print_carnet_cpd(car_model)


if __name__ == '__main__':
    main()
