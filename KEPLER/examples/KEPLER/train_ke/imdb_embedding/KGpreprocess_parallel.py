# coding=utf-8
"""do negative sampling and dump training data"""
import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
CORE_NUMBER = mp.cpu_count()

parser=argparse.ArgumentParser()
parser.add_argument("--dumpPath", type=str, help="path to store output files, do NOT create it previously")
parser.add_argument("-ns", "--negative_sampling_size", type=int, default=1)
parser.add_argument("--train", type=str, help="file name of training triplets")
parser.add_argument("--valid", type=str, help="file name of validation triplets")
parser.add_argument("--ent_desc", type=str, help="path to the entity description file (after BPE encoding)")

def getTriples(path):
    res=[]
    with open(path, "r") as fin:
        lines=fin.readlines()
        for l in tqdm(lines, desc = 'Get Triples Progress Bar'):
            tmp=[int(x) for x in l.split()]
            res.append((tmp[0],tmp[2],tmp[1]))
    return res
def result(res):
    print(res)
def count_frequency(triples, start=4):
    count = {}
    for head, relation, tail in tqdm(triples, desc = 'Count frequency Progress Bar'):
        hr=",".join([str(head), str(relation)])
        tr=",".join([str(tail), str(-relation-1)])
        if hr not in count:
            count[hr] = start
        else:
            count[hr] += 1
        if tr not in count:
            count[tr] = start
        else:
            count[tr] += 1
    return count

def get_true_head_and_tail(triples):
    true_head = {}
    true_tail = {}

    for head, relation, tail in tqdm(triples, desc = 'True head/tail Progress Bar'):
        if (head, relation) not in true_tail:
            true_tail[(head, relation)] = []
        true_tail[(head, relation)].append(tail)
        if (relation, tail) not in true_head:
            true_head[(relation, tail)] = []
        true_head[(relation, tail)].append(head)

    for relation, tail in true_head:
        true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
    for head, relation in true_tail:
        true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))
    return true_head, true_tail

def getTokens(s):
    return min(len(s.split()),512)

# def genSample(id, triples, args, split, Qdesc, true_head, true_tail):
def genSample(data):
    try:
        id = data["id"]
        triples = data["data"]
        args = data["args"]
        split = data["split"]
        Qdesc = data["Qdesc"]
        true_head = data["true_head"]
        true_tail = data["true_tail"]
        print("Spawned sample",split," with id",str(id))
        fHead = open(os.path.join(args.dumpPath, "head", split+"_par_"+ str(id))+".bpe", "w")
        fTail = open(os.path.join(args.dumpPath, "tail", split+ "_par_"+str(id))+".bpe", "w")
        fnHead = open(os.path.join(args.dumpPath, "negHead", split+ "_par_"+str(id))+".bpe", "w")
        fnTail = open(os.path.join(args.dumpPath, "negTail", split+ "_par_"+str(id))+".bpe", "w")
        rel=[]
        sizes=[]
        nE=len(Qdesc)
        #for h, r, t in tqdm(triples, desc = 'Gen sample Progress Bar of pid '+str(id)):
        i = 0
        for h, r, t in triples:
            if not (i%len(triples)):
                print("Pid",str(id),"Reached index:",str(i),"on len:",str(len(triples)))
            i+=1
        
            rel.append(r)
            fHead.write(Qdesc[h])
            fTail.write(Qdesc[t])
            size=getTokens(Qdesc[h])
            size+=getTokens(Qdesc[t])
            for mode in ["head-batch","tail-batch"]:
                negative_sample_list = []
                negative_sample_size = 0
                while negative_sample_size < args.negative_sampling_size:
                    negative_sample = np.random.randint(nE, size=args.negative_sampling_size*2)
                    if mode == 'head-batch':
                        mask = np.in1d(
                            negative_sample,
                            true_head[(r, t)],
                            assume_unique=True,
                            invert=True
                        )
                    elif mode == 'tail-batch':
                        mask = np.in1d(
                            negative_sample,
                            true_tail[(h, r)],
                            assume_unique=True,
                            invert=True
                        )
                    negative_sample = negative_sample[mask]
                    negative_sample_list.append(negative_sample)
                    negative_sample_size += negative_sample.size
                negative_sample = np.concatenate(negative_sample_list)[:args.negative_sampling_size]
                for t in negative_sample:
                    x=int(t)
                    if mode == 'head-batch':
                        fnHead.write(Qdesc[x])
                    else:
                        fnTail.write(Qdesc[x])
                    size += getTokens(Qdesc[x])
            sizes.append(size)
            
        fHead.close()
        fTail.close()
        fnHead.close()
        fnTail.close()
    except Exception as e:
            print(e)
    np.save(os.path.join(args.dumpPath, "relation", split+ "_par_"+str(id))+".npy", np.array(rel))
    np.save(os.path.join(args.dumpPath, "sizes", split+ "_par_"+str(id))+".npy", np.array(sizes))

i = 0
def get_id():
    global i
    i+=1
    return str(i)

def run_sub_process(data, sub_process, args, split, Qdesc, true_head, true_tail):
        from functools import partial
        print("Core number",str(CORE_NUMBER))
        step_size = len(data) // (CORE_NUMBER-1)
        pool = mp.Pool(CORE_NUMBER)
        global i
        i = 0

        sub_data = [ {"id":get_id(), "data":data[index:index+step_size], "args": args, "split": split, "Qdesc": Qdesc, "true_head": true_head, "true_tail": true_tail} for index in range(0, len(data), step_size) ]
        print("LEN OF SUB DATA",len(sub_data))
        # fun = partial(sub_process, args=args, split=split, Qdesc=Qdesc, true_head=true_head, true_tail=true_tail)
        print("Starting sub process:",sub_process,"data len",str(len(sub_data)))
        pool.map(sub_process, sub_data)

        # result = pool.staramp_async(fun, iterable=sub_data)
        #pool.starmap(sub_process, sub_data)
        # pool.close()
        # pool.join()

if __name__=='__main__':
    args=parser.parse_args()
    TrainTriples = getTriples(args.train)
    ValidTriples = getTriples(args.valid)
    AllTriples = TrainTriples + ValidTriples
    Qdesc=[]
    with open(args.ent_desc, "r") as fin:
        Qdesc=fin.readlines()
    print(str(datetime.now())+" load finish")
    print(Qdesc[0])
    count = count_frequency(AllTriples)
    true_head, true_tail = get_true_head_and_tail(AllTriples)
    os.mkdir(args.dumpPath)
    json.dump(count, open(os.path.join(args.dumpPath, "count.json"), "w"))
    for nm in ["head","tail","negHead","negTail","relation","sizes"]:
        os.mkdir(os.path.join(args.dumpPath, nm))
    print(str(datetime.now()) + " preparation finished")
    run_sub_process(TrainTriples, genSample, args, "train", Qdesc, true_head, true_tail)
    # genSample(TrainTriples, args, "train", Qdesc, true_head, true_tail)
    print(str(datetime.now())+" training set finished")
    run_sub_process(TrainTriples, genSample, args, "train", Qdesc, true_head, true_tail)
    # genSample(ValidTriples, args, "valid", Qdesc, true_head, true_tail)
    print(str(datetime.now())+" all finished")



