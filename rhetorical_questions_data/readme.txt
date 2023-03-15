https://github.com/pujan003/Rhetorical-Question-Detection


Non-/Rhetorical Questions Dataset
(The questions have been extracted from Switchboard Dialog Act Corpus: see http://web.stanford.edu/~jurafsky/ws97/ for details)


FORMAT
{0,1},utterance_i {%,&} utterance_i+1 {%,&}-1 utterance_i-1 {%,&}-2 utterance_i-2

FORMAT-DETAILS
0: non-rhetorical
1: rhetorical

&, &-1, &-2 : The given utterance is spoken by the same person who spoke the main utterance.
%, %-1, %-2 : otherwise.

t_con: connective
t_laugh: laugh
t_empty: no utterance (e.g. "% t_empty" means no utterance follows the main utterance.)

EXAMPLE
X: is that right?
Y: uh-huh
X: well,  do you see that there's a big problem with electronic things?
Y: no,

Suppose the third utterance is the main(or target) utterance and is non-rhetorical.
This is encoded in the dataset as:
0, well,  do you see that there's a big problem with electronic things? % no, %-1 uh-huh  &-2 is that right?

CITATION
1. This paper:
@InProceedings{bhattasali-EtAl:2015:ACL-IJCNLP, 
author = {Bhattasali, Shohini and Cytryn, Jeremy and Feldman, Elana and Park, Joonsuk}, 
title = {Automatic Identification of Rhetorical Questions}, 
booktitle = {Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)}, 
month = {July}, 
year = {2015}, 
address = {Beijing, China}, 
publisher = {Association for Computational Linguistics}, 
pages = {743--749}, 
url = {http://www.aclweb.org/anthology/P15-2122} }

2. Please cite the original Switchboard Dialog Act Corpus as well:
@techreport{jurafsky97switchboard,
  added-at = {2009-11-25T18:41:14.000+0100},
  author = {Jurafsky, D. and Shriberg, E. and Biasca, D.},
  biburl = {http://www.bibsonomy.org/bibtex/25d7488032179f3d1c2946689123091a8/sonntag},
  description = {Daniel Sonntag all references},
  institution = {University of Colorado, Institute of Cognitive Science},
  interhash = {cccd61f67e2f6dac4f591898931dabfd},
  intrahash = {5d7488032179f3d1c2946689123091a8},
  key = {Technical Report 97-01},
  keywords = {QABook imported},
  number = {Draft 13},
  owner = {sonntag},
  timestamp = {2009-11-25T18:41:14.000+0100},
  title = {{Switchboard SWBD-DAMSL shallow-discourse-function annotation coders
	manual}},
  year = 1997
}
