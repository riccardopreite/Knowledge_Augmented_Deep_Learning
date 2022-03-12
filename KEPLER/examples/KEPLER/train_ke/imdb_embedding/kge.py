import torch
from torch import cuda
from torch.optim import Adam

from fairseq.models.roberta import RobertaModel
from torchkge.models import TransEModel
from torchkge.models.interfaces import Model

from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
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
lr = 0.000007
n_epochs = 10
b_size = 64 #32768
margin = 0.2
args = roberta.args
for key in roberta:
    print(key)
    print(roberta[key])

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

sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')

iterator = tqdm(range(n_epochs), unit='epoch')
for epoch in iterator:
    running_loss = 0.0
    for i, batch in enumerate(dataloader):
        h, t, r = batch[0], batch[1], batch[2]
        n_h, n_t = sampler.corrupt_batch(h, t, r)
        optimizer.zero_grad()
        # forward + backward + optimize
        pos, neg = model.forward(h, t, n_h, n_t, r)
        loss = criterion(pos, neg)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    iterator.set_description(
        'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                              running_loss / len(dataloader)))
#    checkpoint_utils.save_checkpoint(args, model, epoch+1, running_loss / len(dataloader))
model.optimizer = optimizer.state_dict()
base = {
            'args': args,
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': running_loss / len(dataloader),
            }
for key in model:
    base[key] = model[key]
torch.save(base, "ke_embed.pt")
#torch.save({
#            'args': args,
#            'epoch': epoch+1,
#            'model_state_dict': model.state_dict(),
#            'optimizer': optimizer.state_dict(),
#            'loss': running_loss / len(dataloader),
#            }, "ke_embed.pt")

#torch.save(model.state_dict(), "ke_embed.pt")
#model.normalize_parameters()
