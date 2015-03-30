import xml.dom.minidom
from btk.xmlfilter import *

def main():
#    logging.basicConfig()
#    logging.getLogger("").setLevel(logging.DEBUG)
    
    # open xml File
    fname = "featDesc.xml"
    dom = xml.dom.minidom.parse(fname)
    filterdata = xmlfeatdesc.XMLFeatDesc(dom)

    # Load filters
    from btk.xmlfilter.filter import FilterFactory
    ff = FilterFactory()
    stringfilter.register(ff)
    
    # build FilterGraph
    filter = filterdata.buildFilter("graph", ff)

    # do some work
    for i in filter:
        print i
        #pass
    
if __name__ == "__main__":
    main()
