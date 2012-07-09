TARG = output/fp

default: $(TARG).pdf

bh = beamer-header.tex

# beamerOpts  = -V theme:$(theme) -V colortheme:$(colortheme) -V outertheme:$(outertheme)
beamerOpts += --include-in-header=$(bh)

# beamerOpts += -V theme:Madrid
# beamerOpts += -V theme:Warsaw
beamerOpts += -V theme:Frankfurt
# beamerOpts += -V theme:Berkeley
# beamerOpts += -V colortheme:albatross

# beamerOpts += --incremental
# beamerOpts += -V toc:true

# beamerOpts += --highlight-style=kate

# Hack to stop pandoc from changing list item indents
untweakIndents = sed -e 's/\\renewcommand{\\@listi}/\\newcommand{\\voot}/'

output:
	mkdir output

output/%.tex: %.md makefile $(bh) output
	pandoc $< -f Markdown+LHS -t beamer $(beamerOpts) | $(untweakIndents) > $@

%.pdf: %.tex makefile backus-fortran.jpg BackusTuringPaperHighlight.png
	pdflatex -output-directory output $<

# %-s5.html: %.md makefile
# 	pandoc -t s5 -s $< -o $@ --standalone

# %-slidy.html: %.md makefile
# 	pandoc -t slidy -s $< -o $@ --standalone --incremental

# showpdf=open
# showpdf=evince
# showpdf=explorer
showpdf = open -a Skim.app

see-handout: output/fp-handout.see

handout: output/fp-handout.pdf

output/%-handout.tex: output/%.tex
	sed -e 's/ignorenonframetext/ignorenonframetext,handout/' < $< > $@
# \documentclass[ignorenonframetext,]{beamer}


see: $(TARG).see

%.see: %.pdf
	${showpdf} $<

clean:
	rm output/*

.PRECIOUS: output/%.tex output/%.pdf output/fp-handout.pdf
