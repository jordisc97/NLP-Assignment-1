from __future__ import division
import numpy as np
import skseq.sequences.discriminative_sequence_classifier as dsc
import skseq.sequences.sequence as seq

class StructuredPerceptronValidation(dsc.DiscriminativeSequenceClassifier):
    """
    Implements an Structured  Perceptron
    """

    def __init__(self,
                 observation_labels,
                 state_labels,
                 feature_mapper,
                 learning_rate=1.0,
                 averaged=True):

        dsc.DiscriminativeSequenceClassifier.__init__(self, observation_labels, state_labels, feature_mapper)
        self.learning_rate = learning_rate
        self.averaged = averaged
        self.params_per_epoch = []
        self.parameters = np.zeros(self.feature_mapper.get_num_features())
        self.fitted = False

    def evaluate_corpus(self, sequences, sequences_predictions):
        """Evaluate classification accuracy at corpus level, comparing with
        gold standard."""
        total = 0.0
        correct = 0.0
        for i, sequence in enumerate(sequences):
            pred = sequences_predictions[i]
            for j, y_hat in enumerate(pred.y):
                if sequence.y[j] == y_hat:
                    correct += 1
                total += 1
        return correct / total

    def average_params(self):
        if self.averaged and len(self.params_per_epoch) > 0:
            new_w = 0
            for old_w in self.params_per_epoch:
                new_w += old_w
            new_w /= len(self.params_per_epoch)
            self.parameters = new_w

    def fit(self, dataset, val_dataset, num_epochs, epochs_before_stopping, dir_to_params):
        """
        Parameters
        ----------

        dataset:
        Dataset with the sequences and tags

        num_epochs: int
        Number of epochs that the model will be trained


        Returns
        --------

        Nothing. The method only changes self.parameters.
        """
        if self.fitted:
            print("\n\tWarning: Model already trained")
        
        best_train_acc = 0
        best_val_acc = float('-inf')
        consecutive_epochs_stopping = 0
        from tqdm.notebook import trange
        for epoch in trange(num_epochs, desc="Fitting", leave=False):
            acc = self.fit_epoch(dataset)

            if self.averaged: self.average_params()
            
            val_acc = self.evaluate_corpus(val_dataset.seq_list, self.viterbi_decode_corpus(val_dataset))
            if val_acc > best_val_acc:
                print("Epoch: %i Train Accuracy: %f Validation Accuracy: %f (new best)" % (epoch, acc, val_acc))
                best_val_acc = val_acc
                best_train_acc = acc
                consecutive_epochs_stopping = 0
                self.save_model(dir_to_params)
                print('Model saved')
            else:
                print("Epoch: %i Train Accuracy: %f Validation Accuracy: %f" % (epoch, acc, val_acc))
                consecutive_epochs_stopping += 1
                print('No increase in validation for %i consecutive epochs' % (consecutive_epochs_stopping))

            if consecutive_epochs_stopping >= epochs_before_stopping:
                print('\nEarly stopping performed at epoch %i, saved model at epoch %i' % (epoch, epoch-epochs_before_stopping))
                print('Train Accuracy for saved model: %f' % (best_train_acc))
                print('Validation Accuracy for saved model: %f' % (best_val_acc))
                break


        self.load_model(dir_to_params)
        print('\nBest model successfully loaded')

        self.fitted = True

    def fit_epoch(self, dataset):
        """
        Method used to train the perceptron for a full epoch over the data

        Parameters
        ----------

        dataset:
        Dataset with the sequences and tags.

        num_epochs: int
        Number of epochs that the model will be trained


        Returns
        --------
        Accuracy for the current epoch.
        """
        num_examples = dataset.size()
        num_labels_total = 0
        num_mistakes_total = 0

        from tqdm.notebook import trange
        for i in trange(num_examples, desc="Epoch", leave=False):
            sequence = dataset.seq_list[i]
            num_labels, num_mistakes = self.perceptron_update(sequence)
            num_labels_total += num_labels
            num_mistakes_total += num_mistakes

        self.params_per_epoch.append(self.parameters.copy())
        acc = 1.0 - num_mistakes_total / num_labels_total
        return acc

    def predict_tags_given_words(self, words):
        sequence =  seq.Sequence(x=words, y=words)
        predicted_sequence, _ = self.viterbi_decode(sequence)
        return predicted_sequence.y

    def perceptron_update(self, sequence):
        """
        Method used to train the perceptron for a single datapoint.

        Parameters
        ----------

        sequence:
        datapoint (sequence)


        Returns
        --------
        num_labels: int


        num_mistakes: int

        Accuracy for the current epoch.
        """
        num_labels = 0
        num_mistakes = 0

        predicted_sequence, _ = self.viterbi_decode(sequence)

        y_hat = predicted_sequence.y

        # Update initial features.
        y_t_true = sequence.y[0]
        y_t_hat = y_hat[0]

        if y_t_true != y_t_hat:
            true_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_true)
            self.parameters[true_initial_features] += self.learning_rate
            hat_initial_features = self.feature_mapper.get_initial_features(sequence, y_t_hat)
            self.parameters[hat_initial_features] -= self.learning_rate

        for pos in range(len(sequence.x)):
            y_t_true = sequence.y[pos]
            y_t_hat = y_hat[pos]

            # Update emission features.
            num_labels += 1
            if y_t_true != y_t_hat:
                num_mistakes += 1
                true_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_true)
                self.parameters[true_emission_features] += self.learning_rate
                hat_emission_features = self.feature_mapper.get_emission_features(sequence, pos, y_t_hat)
                self.parameters[hat_emission_features] -= self.learning_rate

            if pos > 0:
                # update bigram features
                # If true bigram != predicted bigram update bigram features
                prev_y_t_true = sequence.y[pos-1]
                prev_y_t_hat = y_hat[pos-1]
                if y_t_true != y_t_hat or prev_y_t_true != prev_y_t_hat:
                    true_transition_features = self.feature_mapper.get_transition_features(
                        sequence, pos-1, y_t_true, prev_y_t_true)
                    self.parameters[true_transition_features] += self.learning_rate
                    hat_transition_features = self.feature_mapper.get_transition_features(
                        sequence, pos-1, y_t_hat, prev_y_t_hat)
                    self.parameters[hat_transition_features] -= self.learning_rate

        pos = len(sequence.x)
        y_t_true = sequence.y[pos-1]
        y_t_hat = y_hat[pos-1]

        if y_t_true != y_t_hat:
            true_final_features = self.feature_mapper.get_final_features(sequence, y_t_true)
            self.parameters[true_final_features] += self.learning_rate
            hat_final_features = self.feature_mapper.get_final_features(sequence, y_t_hat)
            self.parameters[hat_final_features] -= self.learning_rate

        return num_labels, num_mistakes

    def save_model(self, dir):
        """
        Saves the parameters of the model
        """
        fn = open(dir + "parameters_bestval.txt", 'w')
        for p_id, p in enumerate(self.parameters):
            fn.write("%i\t%f\n" % (p_id, p))
        fn.close()

    def load_model(self, dir):
        """
        Loads the parameters of the model
        """
        fn = open(dir + "parameters_bestval.txt", 'r')
        for line in fn:
            toks = line.strip().split("\t")
            p_id = int(toks[0])
            p = float(toks[1])
            self.parameters[p_id] = p
        fn.close()
