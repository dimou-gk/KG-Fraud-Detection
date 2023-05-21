# Skroutz
## This repository contains my work for my Bachelor's thesis. The main idea behind it to create a Fraud detection (FD) system utilizing a Knowledge graph (KG) with the assistance of ensemble learning. ##

**Components:**

1) thesisTradML, which contains the inital experiments done with simple ML algorithms without the KG or ensemble learning. This is purely a first attempt to compare it with the results of the finalized model.
2) Centralities. This file contains code to connect into a local Neo4j instance to get access into centrality measures provided by the Knowledge Graph of the initial dataset. Then training occurs on the dataset with the addition of the centrality measures which were evaluated to be useful as extra features. 
