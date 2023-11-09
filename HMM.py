import random
import argparse
import codecs
import os
import numpy as np


# observations
def process_observation_file(filename):
    try:
        with open(filename, 'r') as file:
            return file.read().strip().split()
    except FileNotFoundError:
        print(f"File {filename} not found.")


class Observation:
    def __init__(self, stateseq, outputseq):
        self.stateseq = stateseq  # sequence of states
        self.outputseq = outputseq  # sequence of outputs

    def __str__(self):
        return ' '.join(self.stateseq) + '\n' + ' '.join(self.outputseq) + '\n'

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return len(self.outputseq)


def read_file_into_dictionary(data_dict, filename):
    try:
        for line in filename:
            tokens = line.strip().split(' ')
            if len(tokens) == 3:
                state1, state2, prob = tokens
                prob = prob.strip()
                if state1 not in data_dict:
                    data_dict[state1] = {}
                data_dict[state1][state2] = prob
    except FileNotFoundError:
        print(f"File {filename} not found.")
    return data_dict


# hmm model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities"""
        ## Both of these are dictionaries of dictionaries. e.g. :
        # {'#': {'C': 0.814506898514, 'V': 0.185493101486},
        #  'C': {'C': 0.625840873591, 'V': 0.374159126409},
        #  'V': {'C': 0.603126993184, 'V': 0.396873006816}}

        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""

        transitions_file = basename + ".trans"
        emissions_file = basename + ".emit"

        try:
            with open(transitions_file, 'r') as trans_file:
                self.transitions = read_file_into_dictionary(self.transitions, trans_file)
        except FileNotFoundError:
            print(f"Unable to open {transitions_file} file")

        try:
            with open(emissions_file, 'r') as emit_file:
                self.emissions = read_file_into_dictionary(self.emissions, emit_file)
        except FileNotFoundError:
            print(f"Unable to open {emissions_file} file")

        return self.transitions, self.emissions

    ## you do this.
    def generate(self, n):
        current_state = "#"  # Start from the initial state
        observed_states = []
        emitted_values = []

        for _ in range(n):
            # Choose the next state based on transition probabilities
            transition_probs = list(map(float, self.transitions[current_state].values()))
            next_state = random.choices(list(self.transitions[current_state].keys()), weights=transition_probs)[0]
            observed_states.append(next_state)

            # Choose an emission value based on emission probabilities
            emission_probs = list(map(float, self.emissions[next_state].values()))
            emitted_value = random.choices(list(self.emissions[next_state].keys()), weights=emission_probs)[0]
            emitted_values.append(emitted_value)

            current_state = next_state

        return Observation(observed_states, emitted_values)

    def forward(self, observation):
        # Get the observed sequence from the observation
        obs_seq = observation.outputseq

        # Store transition and emission probabilities
        transition_probs = self.transitions
        emission_probs = self.emissions

        # Retrieve the list of states and the number of states and observations
        states = list(transition_probs.keys())
        num_states = len(states)
        num_obs = len(obs_seq)

        # Initialize a forward matrix to store intermediate probabilities
        forward_matrix = {state: [0] * len(obs_seq) for state in transition_probs if state != "#"}

        # Initialize initial probabilities based on the observation
        for initial_state in transition_probs["#"]:
            forward_matrix[initial_state][0] = float(transition_probs["#"][initial_state]) * float(
                emission_probs[initial_state].get(obs_seq[0], 0))

        # Propagate forward to compute probabilities for subsequent time steps
        for time_step in range(1, num_obs):
            for next_state in forward_matrix:
                total_prob = 0

                for current_state in forward_matrix:
                    # Compute the probability using the forward algorithm
                    total_prob += float(forward_matrix[current_state][time_step - 1]) * float(
                        transition_probs[current_state].get(next_state, 0)) * float(
                        emission_probs[next_state].get(obs_seq[time_step], 0))

                forward_matrix[next_state][time_step] = total_prob

        # Calculate the final forward probability as the sum of probabilities in the last time step
        final_fwd_prob = sum(forward_matrix[state][-1] for state in forward_matrix)

        # Find the state with the highest probability at the last time step
        max_prob = -1
        final_state = None

        for state in forward_matrix:
            last_element = forward_matrix[state][-1]

            if last_element > max_prob:
                final_state = state
                max_prob = last_element

        # Return the final state and the overall forward probability
        return final_state, final_fwd_prob

    ## you do this: Implement the Viterbi algorithm. Given an Observation (a list of outputs or emissions)
    ## determine the most likely sequence of states.

    def viterbi(self, observation):
        """given an observation,
        find and return the state sequence that generated
        the output sequence, using the Viterbi algorithm.
        """
        obs_seq = observation.outputseq

        T = self.transitions
        E = self.emissions

        states = list(T.keys())
        num_obs = len(obs_seq)

        # Small smoothing constant for Laplace smoothing
        smoothing_constant = 1e-10

        # Initialize the Viterbi path and probabilities
        viterbi = {state: [0] * len(obs_seq) for state in T if state != "#"}
        path_tracker = {state: [0] * len(obs_seq) for state in T if state != "#"}

        # Initialize the probabilities for the initial state
        for state in T["#"]:
            viterbi[state][0] = np.log(float(T["#"][state]) + smoothing_constant) + np.log(float(E[state].get(obs_seq[0], smoothing_constant)))

        # Viterbi algorithm
        for t in range(1, num_obs):
            for next_state in viterbi:
                prob_path_tracker = [(viterbi[current_state][t - 1] + np.log(
                    float(T[current_state].get(next_state, smoothing_constant))) + np.log(
                    float(E[next_state].get(obs_seq[t], smoothing_constant))),
                                      current_state) for current_state in viterbi]
                max_prob, previous_state = max(prob_path_tracker, key=lambda x: x[0])
                viterbi[next_state][t] = max_prob
                path_tracker[next_state][t] = previous_state

        # Find the best path
        best_path_prob, last_state = max((viterbi[state][num_obs - 1], state) for state in viterbi)
        best_path = [last_state]

        for t in range(num_obs - 1, 0, -1):
            last_state = path_tracker[last_state][t]
            best_path.insert(0, last_state)

        return best_path, best_path_prob


def run_hmm(args):
    hmm = HMM()
    hmm.load("partofspeech.browntags.trained")

    if args.forward:
        observation_sequence = process_observation_file(args.forward)
        observation = Observation(stateseq=[], outputseq=observation_sequence)
        fwd_prob = hmm.forward(observation)
        print(f"The final states are: {fwd_prob[0]} and have probability {fwd_prob[1]}")

    if args.viterbi:
        observation_sequence = process_observation_file(args.viterbi)
        observation = Observation(stateseq=[], outputseq=observation_sequence)
        viterbi_sequence = hmm.viterbi(observation)
        print(f"Viterbi's best sequence: {viterbi_sequence}")

    if args.generate:
        generated_sequence = hmm.generate(args.generate)
        print(f"The generated sequence is: {generated_sequence}")


def main():
    parser = argparse.ArgumentParser(description='HMM Arguments')
    parser.add_argument('--generate', type=int)
    parser.add_argument('--forward', type=str)
    parser.add_argument('--viterbi', type=str)

    args = parser.parse_args()

    hmm = HMM()

    hmm.load("partofspeech.browntags.trained")

    if args.forward:
        observation_sequence = process_observation_file(args.forward)
        observation = Observation(stateseq=[], outputseq=observation_sequence)
        fwd_prob = hmm.forward(observation)
        print(f"The final states are: {fwd_prob[0]} and has probability {fwd_prob[1]}")

    if args.viterbi:
        observation_sequence = process_observation_file(args.viterbi)
        observation = Observation(stateseq=[], outputseq=observation_sequence)
        viterbi_sequence = hmm.viterbi(observation)
        print(f"Viterbi's best sequence: {viterbi_sequence}")

    if args.generate:
        generated_sequence = hmm.generate(args.generate)
        print(f"The generated sequence is: {generated_sequence}")


if __name__ == '__main__':
    main()
