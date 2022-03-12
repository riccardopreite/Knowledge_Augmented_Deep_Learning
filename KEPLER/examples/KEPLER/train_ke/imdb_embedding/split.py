import copy, random
f = open("relation.tsv","r")
print("Relation opened")
lines = f.readlines()
length = len(lines)
copied = lines.copy()
print("Readed lines",length)
valid_size = int(length*0.025)
valid_list = []
list_index = []

for i in range(0,valid_size):
   if not (i % 100000):
       print("Reached",str(i))
   valid_list.append(copied.pop(random.randrange(len(copied))))

#valid_list = random.sample(lines, valid_size)
print("Done sampled",str(len(valid_list)))
#for elem in valid_list:
#    copied.remove(elem)
#copied = [x for x in lines if x not in valid_list]
print("Done copied",str(len(copied)))

#while len(valid_list) < valid_size:
#    index = randrange(length)
#    if index not in list_index:
#        l = lines[index]
#        copied.remove(l)
#        list_index.append(index)
#        valid_list.append(l)
train = open("relation_train.txt","w+")
valid = open("relation_valid.txt","w+")

train.writelines(copied)
valid.writelines(valid_list)
