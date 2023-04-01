

def writeToReport(path,content):
    f= open(path,"a")    
    f.write(content+'\n')
    f.close()