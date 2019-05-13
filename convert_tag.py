import os

def editDistance(s1, s2):
    """最小编辑距离"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

if __name__ == '__main__':
    data_dir='/data/share/liuchang/comments_dayu/tag_prediction'
    with open(os.path.join(data_dir,'laosiji_tags_vocab.txt'), 'r', encoding='utf8') as f:
        lsj_tags = f.read().splitlines()
    with open(os.path.join(data_dir,'dayu_tags_vocab.txt'), 'r', encoding='utf8') as f:
        dayu_tags = f.read().splitlines()
    distences=[]
    for i in range(len(dayu_tags)):
        tag=dayu_tags[i]
        dis=[(s,editDistance(tag,s)) for s in lsj_tags]
        dis.sort(key=lambda n:n[1])
        distences.append('\t'.join([n[0] for n in dis[:20]]))

    with open(os.path.join(data_dir, 'dayu_tags.tsv'), 'w', encoding='utf8') as f:
        for t,d in zip(dayu_tags,distences):
            f.write(t+'\t\t'+d+'\n')

