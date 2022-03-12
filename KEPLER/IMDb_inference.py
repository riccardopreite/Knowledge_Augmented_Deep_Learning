from fairseq.models.roberta import RobertaModel
import sys
checkpoint_dir = sys.argv[1]

roberta = RobertaModel.from_pretrained(
    checkpoint_dir,
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='IMDb-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.target_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/IMDb/dev.tsv') as fin:
    #fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sentence, target = tokens[1], tokens[0]
        tokens = roberta.encode(sentence)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy: ', checkpoint_dir, (float(ncorrect)/float(nsamples))*100)
