from datasets import load_dataset

# Load the dataset "heegyu/namuwiki-extracted"
dataset = load_dataset("heegyu/namuwiki-extracted")

# print(dataset['train'][0])
# e.g) {
# 'title': '!!아앗!!',
# 'text': '！！ああっと！！ ▲신 세계수의 미궁 2에서 뜬 !!아앗!! 세계수의 미궁 시리즈에 전통으로 등장하는 대사. 2편부터 등장했으며 훌륭한 사망 플래그의 예시이다. 세계수의 모험가들이 탐험하는 던전인 수해의 구석구석에는 채취/벌채/채굴 포인트가 있으며, 이를 위한 채집 스킬에 ...',
# 'contributors': '110.46.34.123,kirby10,max0243,218.54.117.149,ruby3141,121.165.63.239,iviyuki,1.229.200.194,anatra95,kiri47,175.127.134.2,nickchaos71,chkong1998,kiwitree2,namubot,huwieblusnow',
# 'namespace': ''
# }

# write the dataset to a text file
for split in dataset.keys():
    with open(f"datasets/{split}.txt", "w", encoding='utf-8') as f:
        for example in dataset[split]:
            f.write(example["title"] + "\n")
            f.write(example["text"] + "\n")
