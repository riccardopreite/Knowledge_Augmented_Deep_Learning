import torch
from torch import cuda
from torch.optim import Adam

from fairseq.models.roberta import RobertaModel
from torchkge.models import TransEModel
from torchkge.models.interfaces import Model

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader, Trainer
from torchkge.utils.datasets import load_fb15k
from tqdm.autonotebook import tqdm
from torchkge.utils.pretrained_models import load_pretrained_transe

from fairseq import checkpoint_utils
import pandas as pd
import torchkge

def my_load():
    graph = pd.read_csv('graph.txt', sep=" ", header=None, index_col=False)
    relation_data = graph.rename({0:'from',1:'rel',2:'to'}, axis=1)
    kg_train = torchkge.data_structures.KnowledgeGraph(df=relation_data)
    import json
    ent2id = open("ent2idx.json","w+")
    json.dump({str(key): value for key, value in kg_train.ent2ix.items()},ent2id)
    rel2id = open("rel2idx.json","w+")
    json.dump({str(key): value for key, value in kg_train.rel2ix.items()},rel2id)
    roberta = RobertaModel.from_pretrained('./', checkpoint_file='ke.pt').cuda() #'mlm_pretrained.pt')
    return kg_train, roberta
# Load dataset
kg_train, roberta = my_load()

# Define some hyper-parameters for training
emb_dim = 512
lr = 0.00007
n_epochs = 10
b_size = 16 #32768
margin = 0.2
args = roberta.args
print(args)

# Define the model and criterion
trans_model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2').cuda()
model = roberta
model.forward = trans_model.forward
criterion = MarginLoss(margin)

# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
trainer = Trainer(model, criterion, kg_train, n_epochs, b_size, optimizer=optimizer, sampling_type='bern', use_cuda='all',)

trainer.run()
#checkpoint_utils.save_checkpoint(args, trainer, {epoch+1}, running_loss / len(dataloader))
