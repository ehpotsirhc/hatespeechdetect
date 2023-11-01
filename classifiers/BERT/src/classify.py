#!/usr/bin/env python3

# BERT Classifier Classification Module
# Christophe Leung
# October 30, 2023

# Some parts of the get_validation_performance_with_labels() and model_train() 
# functions have been adapted from the assignment-provided code from CS678.
# Tested on Python 3.10.7; should be compatible between v3.6 v3.11

# -----------------------------------------------------------------------------
from pathlib import Path
from helpers2 import Helpers2 as hlp
from torch.optim import AdamW
from sklearn.metrics import precision_score, recall_score, f1_score
from config import Constants
from preproc import Preproc
import pandas as pd, numpy as np, torch, os

# -----------------------------------------------------------------------------
# Model Analysis
# function to get validation accuracy with prediction-vs-true labels returned
def get_validation_performance_with_labels(val_set, **kwargs):
    hlp.seed_everything()
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 50
    model = Constants.MODEL
    model.eval()    # Put the model in evaluation mode
    total_eval_accuracy, total_eval_loss, total_correct = 0, 0, 0   # Tracking variables 
    labels_pred, labels_true = [], []
    
    num_batches = int(len(val_set)/batch_size) + 1
    for i in range(num_batches):
        end_index = min(batch_size * (i+1), len(val_set))
        batch = val_set[i*batch_size:end_index]
        if len(batch) == 0: continue
        
        input_id_tensors = torch.stack([data[0] for data in batch])
        input_mask_tensors = torch.stack([data[1] for data in batch])
        label_tensors = torch.stack([data[2] for data in batch])
        
        # Move tensors to the GPU
        b_input_ids = input_id_tensors.to(Constants.GPU_DEVICE)
        b_input_mask = input_mask_tensors.to(Constants.GPU_DEVICE)
        b_labels = label_tensors.to(Constants.GPU_DEVICE)
            
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # Note: this line of code might need to change depending on the model
            # the current line will work for bert-base-uncased
            # please refer to huggingface documentation for other models
            outputs = model(b_input_ids, \
                token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits
                
            # Accumulate the validation loss.
            total_eval_loss += loss.item()
            
            # Move logits and labels to CPU
            logits = (logits).detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the number of correctly labeled examples in batch
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = np.argmax(label_ids, axis=1).flatten()

            labels_pred = np.concatenate((labels_pred, pred_flat))
            labels_true = np.concatenate((labels_true, labels_flat))

            num_correct = np.sum(pred_flat == labels_flat)
            total_correct += num_correct
        
    # Report the final accuracy for this validation run.
    labels_pred = labels_pred.astype('int')#+1
    avg_val_accuracy = total_correct / len(val_set)
    # print('labels_pred:', labels_pred)
    # print('labels_true:', labels_true)
    print('Predictions Correct: %s/%s' % (total_correct, len(labels_pred)))
    print('Predictions Expected:', len(val_set))
    return {'avg_val_accuracy': avg_val_accuracy, 'labels_pred':labels_pred, 'labels_true':labels_true}

# -----------------------------------------------------------------------------
# Model Related Functions

# trains a model given a pre-trained model, training set and validation set
def model_train(model, train_set, val_set, **kwargs):
    hlp.separator(msg='Beginning training loop...')
    hlp.seed_everything()
    
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 50
    optimizer = kwargs['optimizer'] if 'optimizer' in kwargs else AdamW(model.parameters())
    epochs = kwargs['epochs'] if 'epochs' in kwargs else 10

    for epoch_i in range(0, epochs):
        # Perform one full pass over the training set.
        print('\n======== Epoch %s / %s ========' % (epoch_i+1, epochs))

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode.
        model.train()

        # For each batch of training data...
        num_batches = int(len(train_set)/batch_size) + 1

        for i in range(num_batches):
            end_index = min(batch_size * (i+1), len(train_set))

            batch = train_set[i*batch_size:end_index]

            if len(batch) == 0: continue

            input_id_tensors = torch.stack([data[0] for data in batch])
            input_mask_tensors = torch.stack([data[1] for data in batch])
            label_tensors = torch.stack([data[2] for data in batch])

            # Move tensors to the GPU
            b_input_ids = input_id_tensors.to(Constants.GPU_DEVICE)
            b_input_mask = input_mask_tensors.to(Constants.GPU_DEVICE)
            b_labels = label_tensors.to(Constants.GPU_DEVICE) 

            optimizer.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # this line of code might need to change depending on the model
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            
            loss = outputs.loss
            logits = outputs.logits

            total_train_loss += loss.item() 

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Update parameters and take a step using the computed gradient.
            optimizer.step()
            
        # Measure the performance on the validation set after the completion of each epoch
        val_acc = get_validation_performance_with_labels(val_set, batch_size=batch_size)
        print('Total loss:', total_train_loss)
        print('Validation accuracy:', val_acc['avg_val_accuracy'])
    print('\nTraining complete!\n\n')


def model_save(model, fpath):
    os.makedirs(Path(fpath).parent) if not Path(fpath).parent.exists() else None
    torch.save(model.state_dict(), fpath)


def model_predict(test_text, **kwargs):
    hlp.seed_everything()
    test_set = Preproc.texts2set(test_text)
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 50
    model = Constants.MODEL
    model.eval()    # Put the model in evaluation mode
    labels_pred = []
    
    num_batches = int(len(test_set)/batch_size) + 1
    for i in range(num_batches):
        end_index = min(batch_size * (i+1), len(test_set))
        batch = test_set[i*batch_size:end_index]
        if len(batch) == 0: continue
        
        input_id_tensors = torch.stack([data[0] for data in batch])
        input_mask_tensors = torch.stack([data[1] for data in batch])
        b_input_ids = input_id_tensors.to(Constants.GPU_DEVICE)
        b_input_mask = input_mask_tensors.to(Constants.GPU_DEVICE)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions; based on bert-based-uncased
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = (outputs.logits).detach().cpu().numpy()

            # Calculate the number of correctly labeled examples in batch
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_pred = np.concatenate((labels_pred, pred_flat))
    
    labels_pred = labels_pred.astype('int')#+1
    print('Predictions Total:', len(labels_pred))
    print('Predictions Expected:', len(test_set))
    return labels_pred


# perform an error analysis on the model and print out the incorrect examples
def model_error(test_text, test_labels_true, **kwargs):
    hyparams = kwargs['hyparams'] if 'hyparams' in kwargs else {}
    n_err = kwargs['n_err'] if 'n_err' in kwargs else 5   # default to 5
    labels_pred = model_predict(test_text, batch_size=hyparams['batch_size'])
    indices_wrong = np.argwhere((labels_pred==test_labels_true) == False).flatten()
    n_correct = (np.array(labels_pred)==np.array(test_labels_true)).sum()
    n_total = len(labels_pred)
    n_correct_percentage = n_correct/n_total*100
    precision = precision_score(test_labels_true, labels_pred, average='micro')
    recall = recall_score(test_labels_true, labels_pred, average='micro')
    f1 = f1_score(test_labels_true, labels_pred, average='micro')
    print('Predictions Correct: %s/%s  (%.3f%%)' % (n_correct, n_total, n_correct_percentage))
    print('Precision/Recall/F1: [%.3f / %.3f / %.3f]' % (precision, recall, f1))

    if indices_wrong.size >= n_err:
        idx_examples = np.random.choice(indices_wrong, size=n_err, replace=False).tolist()
        labels_pred = labels_pred.take(idx_examples).tolist()
        test_labels_true = test_labels_true.take(idx_examples).tolist()
        examples = np.array(test_text).take(idx_examples).tolist()
        combined = list(zip(idx_examples, labels_pred, test_labels_true, examples))
        return [{'index':e[0], 'label_pred':hlp.id2label(e[1]), 'label_true':hlp.id2label(e[2]), 'text':e[3]} for e in combined]
    else:
        return 'There are less than five incorrect examples.'


# -----------------------------------------------------------------------------
# Final Testing/Prediction Processing

# make the final prediciton on the unlabeled test set
def predict_final(df_testing, **kwargs):
    hyparams = kwargs['hyparams'] if 'hyparams' in kwargs else {}
    pred_ids = model_predict(df_testing.sentence.values, batch_size=hyparams['batch_size'])
    pred_labels = [hlp.id2label(pred)[0] for pred in pred_ids]
    return (pred_ids, pred_labels)

# save the unlabeled dataframe
def output_final(predictions_final, fpath_jsonout):
    df_results = pd.DataFrame()
    df_results['pred_id'], df_results['pred_label'] = predictions_final
    df_results.to_csv(Path(fpath_jsonout), index=False)

