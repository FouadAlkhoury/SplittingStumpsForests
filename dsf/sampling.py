import json
import random
import sys

variant = "NoLeafEdges"
pattern_max_size = 6
jsonPath = "../tmp/snippets/"
samplesPath = "../tmp/samples/"
patternsPath = "../frequentTreesInRandomForests/InferenceComparison"
patternList=[]


def sampleRandomForest(dataset, samples_count, patternsCount, max_depth):    

    with open(jsonPath+dataset+'/RF_'+str(max_depth)+'_t1.json') as json_file:
         data = json.load(json_file)
         for c in range(0,samples_count):
                
                patternList=[]
                for i in range(0,patternsCount-1):
                   print(str(data[random.randint(0,len(data)-1)]).replace("'", '"'))
                   patternList.append(str(data[random.randint(0,len(data)-1)]).replace("'", '"')+',')
                patternList.append(str(data[random.randint(0,len(data)-1)]).replace("'", '"')+']')
                f= open(samplesPath+dataset+'/RF_'+str(max_depth)+'_t1_'+str(c+1)+'.json',"w")
                f.write('[')
                for pattern in patternList:
                    
                    f.write(str(pattern)+'\n')
                   
                f.close()            
        
            
    json_file.close()

    
    
dataset = sys.argv[1]
samples_count = 10
patternsCount = int(sys.argv[3])
max_depth = int(sys.argv[2])   
sampleRandomForest(dataset,samples_count, patternsCount, max_depth)