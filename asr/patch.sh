for i in libs/arch.*/*py ; do
    	rm -f $i.tmp
	sed '
	    s,^import common,from btk import common,g
	    s,^import stream,from btk import stream,g
	    s,^import feature,from btk import feature,g
	' $i > $i.tmp 
	mv $i.tmp $i
done
