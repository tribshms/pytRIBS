# GP4
Generic Pre-and-Post Processing Package (GP4) is a python packaged intended to aid in visualizing, analyzing, and trouble-shooting pre-and-post processing steps for a generic distributed hydrologic model (DHM). We focus
here on a generic frame-work as any given distributed hydrologcial models share a number of key similarities that can be capatilzed on to facilitate easier and more efficent pre-and-post processing of model results, regardless of the actual model.
For example, most DHM depend on forcing that varies temporally and spatially. Likewise, most models produce results that represent temporal and spatially varying fluxes and changes of water and energy storage. Consequently, we present two classes that capitalize on these general features: a preprocess class and results class.
Both classes exhibit a high-level of flexibility to accomodate different naming formats and file types as well as provide useful functions for visualizing, analyzing and evaluating pre-and-post process stages in distributed hydrologcical model simulations.

### TODO
Need to node/element subclass to assign individual properties to them, rather have a list.
