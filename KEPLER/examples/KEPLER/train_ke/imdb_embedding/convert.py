import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, help="path to original text file")
parser.add_argument("--train", type=str, help="path to original training data file")
parser.add_argument("--valid", type=str, help="path to original validation data file")
parser.add_argument("--converted_text", type=str, default="Qdesc.txt", help="path to converted text file")
parser.add_argument("--converted_train", type=str, default="train.txt", help="path to converted training file")
parser.add_argument("--converted_valid", type=str, default="valid.txt", help="path to converted validation file")
Cnt=0
if __name__=='__main__':
    args = parser.parse_args()
    Qid={}  #Entity to id (line number in the description file)
    Pid={}  #Relation to id
    def getNum(s):
        return int(s[1:])
    with open(args.text, "r") as fin:
        with open(args.converted_text, "w") as fout:
            lines = fin.readlines()
            for idx, line in enumerate(lines):
                data = line.strip().split('\t')
                if not len(data) >= 2:
                    continue
                id = data[0]
                #splitted = data[0].split("_")
                #print(data)
                #desc = '\t'.join(splitted[1]).strip()
                desc = data[1].replace("\n","")
                print(desc)
                fout.write(desc+"\n")
                Qid[id] = Cnt#idx
                Cnt+=1
    def convert_triples(inFile, outFile):
        Cnt = max(Qid.values())
        with open(inFile, "r") as fin:
            with open(outFile, "w") as fout:
                lines = fin.readlines()
                for line in lines:
                    data = line.strip().split('\t')
                    if len(data) != 3:
                        print(data)
                    assert len(data) == 3
                    
                    if data[1] not in Pid:
                        Pid[data[1]] = len(Pid)
                    if data[0] not in Qid.keys():
                        id = data[0]
                        splitted = data[0].split("_")
                        desc = splitted[1]
                        print(id+"\t"+desc)
                        import os
                        os.system("echo "+id+"\t"+desc+">> entity.txt")
                        Qid[data[0]] = Cnt
                        Cnt+=1
                        print("added",id,"since was not in entity with Cnt",str(Cnt))
                    fout.write("%d %d %d\n"%(Qid[data[0]], Pid[data[1]], Qid[data[2]]))
    convert_triples(args.train, args.converted_train)
    convert_triples(args.valid, args.converted_valid)
