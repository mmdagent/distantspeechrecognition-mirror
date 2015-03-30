for i in libs/arch.*/*py ; do
    	rm -f $i.tmp
	sed '
	    s,^import stream,from sfe import stream,g
	    s,^import feature,from sfe import feature,g
	' $i > $i.tmp 
	mv $i.tmp $i
done
