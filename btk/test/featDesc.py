import xml.dom.minidom

def handleFeatDesc(featDesc):
    banks = featDesc.getElementsByTagName("filterbank")
    return handleFilterBanks(banks)

def handleFilterBanks(banks):
    bs = {}
    for bank in banks:
        (name, b) = handleFilterBank(bank)
        bs[name] = b
    return bs

def handleFilterBank(bank):
    f = []
    for node in bank.childNodes:
        if node.nodeType == node.ELEMENT_NODE:
            nodeType = node.nodeName 
            if nodeType == "filter":
                f.append(handleFilter(node))
    name = bank.getAttribute("name")
    return (name, f)

def handleFilter(filter):
    params = filter.getElementsByTagName("param")
    ps = handleParams(params)
    nexts = filter.getElementsByTagName("next")    
    n = handleNexts(nexts)
    f = {}
    f["name"] = filter.getAttribute("name")
    f["type"] = filter.getAttribute("type")
    f["params"] = ps
    f["next"] = n
    return f

def handleParams(params):
    ps = {}
    for param in params:
        p = handleParam(param)
        ps[p[0]] = p[1]
    return ps

def handleParam(param):
    name = param.getAttribute("name")
    value = param.getAttribute("value")
    return (name, value)

def handleNexts(nexts):
    n = []
    for next in nexts:
        n.append(handleNext(next))
    return n

def handleNext(next):
    return next.getAttribute("name")

def mergeDicts(dict1, dict2):
    retval = dict2
    for (k,v) in dict1.iteritems():
        retval[k] = v
    return retval

def buildFilter(filters, ff, dict={}):
    source = filters[0]
    parent = ff.produce(source["type"])()
    parent.setParams(mergeDicts(dict, source["params"]))
    for filter in filters[1:]:
        parent = ff.produce(filter["type"])(parent)
        parent.setParams(mergeDicts(dict, filter["params"]))
    return parent
        
if __name__ == "__main__":
    fname = "featDesc.xml"
    dom = xml.dom.minidom.parse(fname)
    fbs = handleFeatDesc(dom)

    import filter
    import btkfilter
    
#    filters = {}
#    for (name, fb) in fbs.iteritems():
#        filters[name] = buildFilter(fb, filter.ff)

#    for i in filters["filter1"]:
#        print i

    dataDir = "../../testdata/sndarray"
    files = []
    for i in range(1,17):
        files.append("%s/e029ach2_061.16.%s.adc.shn" %(dataDir, i))

    beam = buildFilter(fbs["beamformer"], filter.ff, {"filelist": files})

    for j in range(10):
        for i in beam:
            #raw_input("Juhu")
            pass
