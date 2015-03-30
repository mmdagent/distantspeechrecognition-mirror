name = millennium-book

html:
	xsltproc -xinclude -o $(name).html /usr/share/sgml/docbook/xsl-ns-stylesheets/xhtml-1_1/docbook.xsl $(name).xml

pdf:
	xsltproc -xinclude -o $(name).fo /usr/share/sgml/docbook/xsl-ns-stylesheets/fo/docbook.xsl $(name).xml
	fop $(name).fo -pdf $(name).pdf

clean:
	rm -f $(name).{html,fo,pdf}

