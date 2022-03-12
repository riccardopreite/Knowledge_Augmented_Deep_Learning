from fairseq.models.roberta import RobertaModel
import sys
checkpoint_dir = sys.argv[1]
filename = sys.argv[2] if len(sys.argv) > 2 else 'checkpoint_best.pt'
roberta = RobertaModel.from_pretrained(
    checkpoint_dir,
    checkpoint_file=filename,
    data_name_or_path='CoLA-bin'
)

label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.target_dictionary.nspecial]
)
ncorrect, nsamples = 0, 0
roberta.cuda()
roberta.eval()
with open('glue_data/CoLA/dev.tsv') as fin:
    fin.readline()
    for index, line in enumerate(fin):
        tokens = line.strip().split('\t')
        sentence, target = tokens[3], tokens[1]
        tokens = roberta.encode(sentence)
        prediction = roberta.predict('sentence_classification_head', tokens).argmax().item()
        prediction_label = label_fn(prediction)
        ncorrect += int(prediction_label == target)
        nsamples += 1
print('| Accuracy of: ', checkpoint_dir, (float(ncorrect)/float(nsamples))*100)
