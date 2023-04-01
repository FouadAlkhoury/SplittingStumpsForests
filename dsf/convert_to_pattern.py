import json
import re

forest = '[{"id":0,"numSamples":10169,"probLeft":0.9203461500639197,"probRight":0.07965384993608024,"isCategorical":"False","feature":58,"split":0,"leftChild":{"id":1,"numSamples":9359,"probLeft":0.21882679773480074,"probRight":0.7811732022651993,"isCategorical":"False","feature":0,"split":29,"leftChild":{"id":2,"numSamples":2048,"probLeft":0.44970703125,"probRight":0.55029296875,"isCategorical":"False","feature":0,"split":23,"leftChild":{"id":3,"numSamples":921,"prediction":[0.9855172413793103,0.014482758620689656]},"rightChild": {"id":4,"numSamples":1127,"prediction":[0.7567723342939481,0.24322766570605187]}},"rightChild": {"id":5,"numSamples":7311,"probLeft":0.3669812611133908,"probRight":0.6330187388866092,"isCategorical":"False","feature":50,"split":0,"leftChild":{"id":6,"numSamples":2683,"prediction":[0.6164672765657987,0.38353272343420125]},"rightChild": {"id":7,"numSamples":4628,"prediction":[0.26001357773251865,0.7399864222674813]}}},"rightChild": {"id":8,"numSamples":810,"probLeft":0.33209876543209876,"probRight":0.6679012345679012,"isCategorical":"False","feature":0,"split":32,"leftChild":{"id":9,"numSamples":269,"probLeft":0.24907063197026022,"probRight":0.7509293680297398,"isCategorical":"False","feature":29,"split":0,"leftChild":{"id":10,"numSamples":67,"prediction":[0.7788461538461539,0.22115384615384615]},"rightChild": {"id":11,"numSamples":202,"prediction":[0.973293768545994,0.026706231454005934]}},"rightChild": {"id":12,"numSamples":541,"probLeft":0.5083179297597042,"probRight":0.49168207024029575,"isCategorical":"False","feature":27,"split":0,"leftChild":{"id":13,"numSamples":275,"prediction":[0.823943661971831,0.176056338028169]},"rightChild": {"id":14,"numSamples":266,"prediction":[0.35910224438902744,0.6408977556109726]}}}}]'
#[{"patternid":0,"pattern":{"id":0,"feature":50,"split":0,"rightChild":{"id":0,"feature":13,"split":0,"leftChild":{"id":0,"feature":10,"split":0}}}},
forest ='[{"id":0,"numSamples":10209,"probLeft":0.20188069350573024,"probRight":0.7981193064942698,"isCategorical":"False","feature":0,"split":28,"leftChild":{"id":1,"numSamples":2061,"probLeft":0.9864143619602135,"probRight":0.013585638039786511,"isCategorical":"False","feature":46,"split":0,"leftChild":{"id":2,"numSamples":2033,"probLeft":0.940973930152484,"probRight":0.05902606984751599,"isCategorical":"False","feature":41,"split":0,"leftChild":{"id":3,"numSamples":1913,"probLeft":0.9043387349712494,"probRight":0.09566126502875065,"isCategorical":"False","feature":39,"split":0,"leftChild":{"id":4,"numSamples":1730,"probLeft":0.9023121387283237,"probRight":0.09768786127167631,"isCategorical":"False","feature":47,"split":0,"leftChild":{"id":5,"numSamples":1561,"prediction":[0.8856676194365047,0.1143323805634953]},"rightChild": {"id":6,"numSamples":169,"prediction":[0.9465648854961832,0.05343511450381679]}},"rightChild": {"id":7,"numSamples":183,"probLeft":0.5027322404371585,"probRight":0.4972677595628415,"isCategorical":"False","feature":51,"split":0,"leftChild":{"id":8,"numSamples":92,"prediction":[0.6153846153846154,0.38461538461538464]},"rightChild": {"id":9,"numSamples":91,"prediction":[0.8562091503267973,0.1437908496732026]}}},"rightChild": {"id":10,"numSamples":120,"probLeft":0.9416666666666667,"probRight":0.058333333333333334,"isCategorical":"False","feature":52,"split":0,"leftChild":{"id":11,"numSamples":113,"probLeft":0.9911504424778761,"probRight":0.008849557522123894,"isCategorical":"False","feature":15,"split":0,"leftChild":{"id":12,"numSamples":112,"prediction":[0.9606741573033708,0.03932584269662921]},"rightChild": {"id":13,"numSamples":1,"prediction":[0.0,1.0]}},"rightChild": {"id":14,"numSamples":7,"prediction":[1.0,0.0]}}},"rightChild": {"id":15,"numSamples":28,"probLeft":0.9642857142857143,"probRight":0.03571428571428571,"isCategorical":"False","feature":63,"split":62,"leftChild":{"id":16,"numSamples":27,"probLeft":0.7037037037037037,"probRight":0.2962962962962963,"isCategorical":"False","feature":13,"split":0,"leftChild":{"id":17,"numSamples":19,"probLeft":0.8421052631578947,"probRight":0.15789473684210525,"isCategorical":"False","feature":58,"split":0,"leftChild":{"id":18,"numSamples":16,"prediction":[0.7241379310344828,0.27586206896551724]},"rightChild": {"id":19,"numSamples":3,"prediction":[0.16666666666666666,0.8333333333333334]}},"rightChild": {"id":20,"numSamples":8,"probLeft":0.875,"probRight":0.125,"isCategorical":"False","feature":51,"split":0,"leftChild":{"id":21,"numSamples":7,"prediction":[0.9090909090909091,0.09090909090909091]},"rightChild": {"id":22,"numSamples":1,"prediction":[1.0,0.0]}}},"rightChild": {"id":23,"numSamples":1,"prediction":[0.0,1.0]}}},"rightChild": {"id":24,"numSamples":8148,"probLeft":0.23699067255768286,"probRight":0.7630093274423171,"isCategorical":"False","feature":59,"split":0,"leftChild":{"id":25,"numSamples":1931,"probLeft":0.9176592439150699,"probRight":0.08234075608493009,"isCategorical":"False","feature":61,"split":4668,"leftChild":{"id":26,"numSamples":1772,"probLeft":0.7827313769751693,"probRight":0.2172686230248307,"isCategorical":"False","feature":29,"split":0,"leftChild":{"id":27,"numSamples":1387,"probLeft":0.8738284066330209,"probRight":0.12617159336697908,"isCategorical":"False","feature":58,"split":0,"leftChild":{"id":28,"numSamples":1212,"prediction":[0.6425221469515373,0.35747785304846275]},"rightChild": {"id":29,"numSamples":175,"prediction":[0.7728937728937729,0.2271062271062271]}},"rightChild": {"id":30,"numSamples":385,"probLeft":0.9688311688311688,"probRight":0.03116883116883117,"isCategorical":"False","feature":23,"split":0,"leftChild":{"id":31,"numSamples":373,"prediction":[0.8791018998272885,0.12089810017271158]},"rightChild": {"id":32,"numSamples":12,"prediction":[0.05263157894736842,0.9473684210526315]}}},"rightChild": {"id":33,"numSamples":159,"probLeft":0.9748427672955975,"probRight":0.025157232704402517,"isCategorical":"False","feature":36,"split":0,"leftChild":{"id":34,"numSamples":155,"probLeft":0.8580645161290322,"probRight":0.14193548387096774,"isCategorical":"False","feature":42,"split":0,"leftChild":{"id":35,"numSamples":133,"prediction":[0.04477611940298507,0.9552238805970149]},"rightChild": {"id":36,"numSamples":22,"prediction":[0.2647058823529412,0.7352941176470589]}},"rightChild": {"id":37,"numSamples":4,"probLeft":0.75,"probRight":0.25,"isCategorical":"False","feature":10,"split":0,"leftChild":{"id":38,"numSamples":3,"prediction":[0.4,0.6]},"rightChild": {"id":39,"numSamples":1,"prediction":[0.0,1.0]}}}},"rightChild": {"id":40,"numSamples":6217,"probLeft":0.2000965095705324,"probRight":0.7999034904294676,"isCategorical":"False","feature":50,"split":0,"leftChild":{"id":41,"numSamples":1244,"probLeft":0.9316720257234726,"probRight":0.06832797427652733,"isCategorical":"False","feature":36,"split":0,"leftChild":{"id":42,"numSamples":1159,"probLeft":0.9016393442622951,"probRight":0.09836065573770492,"isCategorical":"False","feature":61,"split":4718,"leftChild":{"id":43,"numSamples":1045,"prediction":[0.6824242424242424,0.31757575757575757]},"rightChild": {"id":44,"numSamples":114,"prediction":[0.027472527472527472,0.9725274725274725]}},"rightChild": {"id":45,"numSamples":85,"probLeft":0.9411764705882353,"probRight":0.058823529411764705,"isCategorical":"False","feature":16,"split":0,"leftChild":{"id":46,"numSamples":80,"prediction":[0.9385964912280702,0.06140350877192982]},"rightChild": {"id":47,"numSamples":5,"prediction":[1.0,0.0]}}},"rightChild": {"id":48,"numSamples":4973,"probLeft":0.34566659963804547,"probRight":0.6543334003619545,"isCategorical":"False","feature":26,"split":9,"leftChild":{"id":49,"numSamples":1719,"probLeft":0.18150087260034903,"probRight":0.8184991273996509,"isCategorical":"False","feature":26,"split":7,"leftChild":{"id":50,"numSamples":312,"prediction":[0.7078189300411523,0.29218106995884774]},"rightChild": {"id":51,"numSamples":1407,"prediction":[0.4129662522202487,0.5870337477797514]}},"rightChild": {"id":52,"numSamples":3254,"probLeft":0.7252612169637369,"probRight":0.27473878303626303,"isCategorical":"False","feature":11,"split":0,"leftChild":{"id":53,"numSamples":2360,"prediction":[0.13827226531158063,0.8617277346884193]},"rightChild": {"id":54,"numSamples":894,"prediction":[0.2781021897810219,0.7218978102189781]}}}}}}]'
jsonforest = json.loads(forest)




#print(subtrees[0])
#print(subtrees[1])

thresholds = [0.55,0.6,0.68,0.8,0.94]
#thresholds = [0.68]
def is_valid(tree, threshold, index):
    global subtreeStr
    #global subtrees
    if ("probLeft" in tree and "leftChild" in tree):
        #print(tree["probLeft"])

        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold):

            #del tree["numSamples"]
            #del tree["isCategorical"]
            if("probLeft" in tree["leftChild"] and "leftChild" in tree):



                leftTree = tree["leftChild"]
                is_valid(leftTree, threshold, index)
            else:
                del tree["leftChild"]
            if ("probRight" in tree["rightChild"] and "rightChild" in tree):
                rightTree = tree["rightChild"]
                is_valid(rightTree, threshold, index)
            else:
                del tree["rightChild"]

        else:

            #subtrees.remove(tree)
            #print('Removed')
            #print(tree)

            #tree["id"] = -1
            #del tree["numSamples"]
            #del tree["probLeft"]
            #del tree["probRight"]
            #del tree["isCategorical"]
            #del tree["feature"]
            #del tree["split"]
            treeLeft = tree["leftChild"]
            treeRight = tree["rightChild"]
            #print(treeLeft)
            #print(treeRight)
            subtrees.append(treeLeft)
            subtrees.append(treeRight)

            #index = subtrees.index(tree)
            #subtrees[index] = ""
            #print('removed index: '+str(index))
            #tree = "new"
            #print(subtrees[index])
            #if("numSamples" in tree):
                #del tree["numSamples"]
            subtreeStr = subtreeStr.replace(str(tree),'')
            #tree = ""

            #subtrees[index] = None
            #print(subtree)

    else:

        subtreeStr = subtreeStr.replace(str(tree), '')

    if (tree is not None and "prediction" in tree):

        subtreeStr = subtreeStr.replace(str(tree), '')
        #subtrees[index] = None
        #tree = ""

def check_subtree(subtree):
    print(subtree)
    while("probLeft" in subtree):
        if(subtree["id"] == -1):
            del subtree
            break
        else:
            leftSubtree = subtree["leftChild"]
            rightSubtree = subtree["rightChild"]
            check_subtree(leftSubtree)
            check_subtree(rightSubtree)

def remove_a_key(d, remove_key):
    if isinstance(d, dict):
        for key in list(d.keys()):


                if key == remove_key:
                    del d[key]
                else:
                    remove_a_key(d[key], remove_key)

def remove_a_child_key(d):
    if isinstance(d, dict):
        for key in list(d.keys()):
            if ((key =='leftChild' or key == 'rightChild') and (len(d[key]) == 0)):


                    del d[key]

            else:
                    remove_a_child_key(d[key])

for th in thresholds:
    patterns = set()
    #global subtrees
    global subtreeStr
    subtrees = []
    print('Threshold: ' + str(th))
    jsonforest = json.loads(forest)

    for jsontree in jsonforest:


        #remove_a_key(jsontree, 'id')
        remove_a_key(jsontree, 'numSamples')
        remove_a_key(jsontree, 'isCategorical')
        remove_a_key(jsontree, 'prediction')
        #remove_a_key(jsontree, 'probLeft')
        #remove_a_key(jsontree, 'probRight')
        #remove_a_child_key(jsontree)

        #remove_a_child_key(jsontree, 'rightChild')
        # remove_a_key(jsontree, 'rightChild')
        # remove_a_key(jsontree, 'rightChild')
        # remove a node if it has only one child (left: id)
        subtrees.append(jsontree)
    for index,subtree in enumerate(subtrees):
        #print(index)
        #print(subtree)
        subtreeStr = str(subtree)
        is_valid(subtree, th, index)
        subtreeStr = re.sub(", 'leftChild': {'id': [0-9]+}", '', subtreeStr)
        subtreeStr = re.sub(", 'leftChild': ,", ',', subtreeStr)
        subtreeStr = re.sub(", 'rightChild': {'id': [0-9]+}", '', subtreeStr)
        subtreeStr = re.sub(", 'rightChild': ,", ',', subtreeStr)
        subtreeStr = re.sub(", 'rightChild': }", '}', subtreeStr)
        subtreeStr = re.sub(", 'leftChild': }", '}', subtreeStr)
        subtreeStr = re.sub("'split'",'"split"',subtreeStr)
        subtreeStr = re.sub("'feature'", '"feature"', subtreeStr)

        subtreeStr = re.sub("'probLeft':(.)+?,", '', subtreeStr)
        subtreeStr = re.sub("'probRight':(.)+?,", '', subtreeStr)

        #subtreeStr = re.sub("'probRight'", '"probRight"', subtreeStr)
        subtreeStr = re.sub("'id'", '"id"', subtreeStr)
        subtreeStr = re.sub("'leftChild'", '"leftChild"', subtreeStr)
        subtreeStr = re.sub("'rightChild'", '"rightChild"', subtreeStr)
        #jsontree = json.loads(subtreeStr)
        #remove_a_key(jsontree, 'leftChild')
        #remove_a_key(jsontree, 'rightChild')
        #subtreeStr = str(jsontree)
        if (len(subtreeStr) > 2):

            patterns.add(subtreeStr)
        #print(subtreeStr)

        subtreeStr =''

    print("patterns:")
    #pruned_file = graph_file[:-5] + '_pruned_' + str(th) + '.json'
    #snippets_file = os.path.join(snippetsPath, dataset, pruned_file)

    #with open(snippets_file, 'w') as f_out:
    #    f_out.write("[")
    print('[')
    for p in patterns:
        print('{"patternid": 0, "pattern": ' + str(p) + '},')
        #print(p)
        #print('}')
    print(']')

'''
    for subtree in subtrees:
        #check_subtree(subtree)
        if (subtree is not None):


            print(subtree)
'''


        #print(jsontree)

#str_output = re.sub('"numSamples(.*?)prediction(.*?)]}', '', forest)
#str_output = re.sub('"numSamples(.*?)rical":"False"', '', str_output)
#print(str_output)
#print(jsontree[0]["leftChild"]["numSamples"])

#jsontree[0]["leftChild"]["numSamples"] = 3
#print(jsontree[0]["leftChild"]["numSamples"])
#del jsontree[0]["numSamples"]
#for key,value in jsontree[0].items():



#print('patterns')
#for p in subtrees:
#    if (p["id"] != -1):
#        print(p)
    #print(p)
'''
def is_valid(tree, threshold, index):
    #global tree
    if ("probLeft" in tree):
        print(tree["probLeft"])

        if (tree["probLeft"] <= threshold and tree["probRight"] <= threshold):

            del tree["numSamples"]
            del tree["isCategorical"]
            if("probLeft" in tree["leftChild"]):



                leftTree = tree["leftChild"]
                is_valid(leftTree, threshold, index)
            else:
                del tree["leftChild"]
            if ("probRight" in tree["rightChild"]):
                rightTree = tree["rightChild"]
                is_valid(rightTree, threshold, index)
            else:
                del tree["rightChild"]

        else:

            #subtrees.remove(tree)
            #print('Removed')
            #print(tree)

            #tree["id"] = -1
            #del tree["numSamples"]
            #del tree["probLeft"]
            #del tree["probRight"]
            #del tree["isCategorical"]
            #del tree["feature"]
            #del tree["split"]
            treeLeft = tree["leftChild"]
            treeRight = tree["rightChild"]
            #print(treeLeft)
            #print(treeRight)
            subtrees.append(treeLeft)
            subtrees.append(treeRight)
            #index = subtrees.index(tree)
            #subtrees[index] = ""
            print('removed index: '+str(index))
            tree = "new"
            print(subtrees[index])
            #subtrees[index] = None
            #print(subtree)


    if (tree is not None and "prediction" in tree):
        subtrees[index] = None


def check_subtree(subtree):
    print(subtree)
    while("probLeft" in subtree):
        if(subtree["id"] == -1):
            del subtree
            break
        else:
            leftSubtree = subtree["leftChild"]
            rightSubtree = subtree["rightChild"]
            check_subtree(leftSubtree)
            check_subtree(rightSubtree)


for th in thresholds:
    subtrees = []
    print('Threshold: ' + str(th))
    jsonforest = json.loads(forest)

    for jsontree in jsonforest:
        subtrees.append(jsontree)
    for index,subtree in enumerate(subtrees):
        print(index)
        print(subtree)
        is_valid(subtree, th, index)

    for subtree in subtrees:
        #check_subtree(subtree)
        if (subtree is not None):


            print(subtree)
            
            
            for size in forest_sizes:
    for depth in forest_depths:
            
        graph_file = 'RF_'+str(size)+'_'+str(depth)+'.json'    
        forests_file = os.path.join(forestsPath, dataset, graph_file)
        #data = json.loads(forests_file)

        print(forests_file)
        with open(forests_file, 'r') as f_decision_forests:
        
            trees = json.load(f_decision_forests)
            
            
            
            for th in edge_thresholds:
                
                start_pruning_time = datetime.datetime.now()

                features = []
                splits = []
                patterns = set()
                patterns_scores = set()
                counter = 0
                score = 0
                scoreStr = ''

                pruned_file = graph_file[:-5] + '_pruned_' +str(th)+ '.json'    
                snippets_file = os.path.join(snippetsPath, dataset, pruned_file)  
                
                with open(snippets_file,'w') as f_out:
                    f_out.write("[")
                    for tree in trees:
                        #subtrees = extract_subtrees(tree)
                        #print(subtrees)
                        print("tree")
                        # pattern
                        pattern = ''
                        cont = True
                        #global suffix
                        suffix = ''
                        #global threshold
                        #threshold = th 
                        #root = Node(tree)
                        #extract_subtrees(tree)
                        #result = extract_patterns(tree)
                        #print(result)

                        traverse(tree, th, 0, '','')
                        #pattern += suffix
                        # patterns_scores.add(pattern+'#'+str(scoreStr)+'\n')

                        pattern = ''
                        suffix = ''
                        index = 0
                    for p in patterns:
                        f_out.write(p)
                        
                     
                        counter += 1
                    if (len(patterns) > 0):

                        f_out.seek(0,2)
                        f_out.seek(f_out.tell() - 2, 0)
                        f_out.truncate()
                        f_out.write("]")
                    
                    end_pruning_time = datetime.datetime.now()
                    pruning_time = (end_pruning_time - start_pruning_time)
                    writeToReport(report_pruning_file, str(size)+ ', ' + str(depth) + ', ' + str(th) + ', ' + str(pruning_time) + '\n')            



'''