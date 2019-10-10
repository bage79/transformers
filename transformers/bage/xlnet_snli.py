"""
Ref: http://mccormickml.com/2019/09/19/XLNet-fine-tuning/
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange

from transformers import AdamW
from transformers.modeling_xlnet import XLNetForSequenceClassification
from transformers.tokenization_xlnet import XLNetTokenizer

use_gpu = False
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    num_batch = 112

    cola = True
    cola_train = "glue_data/CoLA/original/tokenized/in_domain_train.tsv"
    cola_dev = "glue_data/CoLA/original/raw/out_of_domain_dev.tsv"

    snli_train = "glue_data/SNLI/train.tsv"
    snli_test = "glue_data/SNLI/test.tsv"

    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True)

    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token

    print(tokenizer.all_special_tokens)

    if cola:
        df = pd.read_csv(cola_train, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
        print(df.head())
        # df.sample(10)

        # BERT: [CLS] + Sentence_A + [SEP] + Sentence_B + [SEP]
        # XLNET: Sentence_A + [SEP] + Sentence_B + [SEP] + [CLS]
        # Create sentence and label lists for cola
        sentences = df.sentence.values
        sentences = [sentence + " <sep> <cls>" for sentence in sentences]  # for single sentence
        labels = df.label.values
        num_labels = 2

        # Inputs
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
        print("Tokenize the first sentence:")
        print(tokenized_texts[0])

    else:
        df_snli = pd.read_csv(snli_train, delimiter='\t', header=0, error_bad_lines=False)
        df_snli = df_snli.dropna()
        # df_snli = df_snli[:10000]

        # Create sentence and label lists for snli
        labels = []
        LABELS = {'contradiction': 0, 'neutral': 1, 'entailment': 2}
        tokenized_texts = []
        for i, row in df_snli.iterrows():
            if row['gold_label'] == "-" or pd.isnull(row['sentence1']) or pd.isnull(row['sentence2']):
                continue

            labels.append(LABELS[row['gold_label']])
            sent1 = tokenizer.tokenize(row['sentence1'])
            sent2 = tokenizer.tokenize(row['sentence2'])
            tokenized_texts.append(sent1 + [sep_token] + sent2 + [sep_token] + [cls_token])

        num_labels = 3

    # Set the maximum sequence length.
    # The longest sequence in our cola training set is 47, snli training set is ,
    # but we'll leave room on the end anyway

    MAX_LEN = 128

    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    _input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad our input tokens
    input_ids = pad_sequences(_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                           random_state=2018, test_size=0.1)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning with XLNet, the authors recommend a batch size of 32, 48, or 128. We will use 32 here to avoid memory issues.
    batch_size = num_batch

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop,
    # with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels=num_labels)
    model.to(device)

    # Param settings
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)


    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10

    # trange is a tqdm wrapper around the normal python range
    for epoch in trange(epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

            print('Epoch: {}\tStep:{}\tLoss: {:.4f}'.format(epoch, step, loss.item()), end='\r')

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                output = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    plt.figure(figsize=(15, 8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    # plt.show()
    plt.savefig('logs/training_loss.png')

    """
    For Dev
    """
    df = pd.read_csv(cola_dev, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    # Create sentence and label lists
    sentences = df.sentence.values

    # We need to add special tokens at the beginning and end of each sentence for XLNet to work properly
    sentences = [sentence + " [SEP] [CLS]" for sentence in sentences]
    labels = df.label.values

    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

    MAX_LEN = 128
    # Use the XLNet tokenizer to convert the tokens to their index numbers in the XLNet vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    batch_size = num_batch

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # Import and evaluate each test batch using Matthew's correlation coefficient
    from sklearn.metrics import matthews_corrcoef

    matthews_set = []

    for i in range(len(true_labels)):
        matthews = matthews_corrcoef(true_labels[i],
                                     np.argmax(predictions[i], axis=1).flatten())
        matthews_set.append(matthews)

    print(matthews_set)

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    final_score = matthews_corrcoef(flat_true_labels, flat_predictions)
    print(final_score)
