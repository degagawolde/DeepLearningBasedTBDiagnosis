	V-??_@V-??_@!V-??_@	x/?L????x/?L????!x/?L????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:V-??_@?MbX9(@A?V?\@YF????x??rEagerKernelExecute 0*	    @??@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2Zd;?O]G@!
!(??X@)Zd;?O]G@1
!(??X@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip1?ZdG@!?w??C?X@)???S㥫?1jN??xw??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice{?G?z??!?C1??ӵ?){?G?z??1?C1??ӵ?:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice?? ?rh??!??vƗ???)?? ?rh??1??vƗ???:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?$??cG@!S?9hr?X@)X9??v???1*a?ڂ???:Preprocessing2F
Iterator::Modely?&1???!e+x?ێ??)y?&1???1e+x?ێ??:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2?~j?t???!{?ԽN1??)?~j?t???1{?ԽN1??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9y/?L????I?3?*?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?MbX9(@?MbX9(@!?MbX9(@      ?!       "      ?!       *      ?!       2	?V?\@?V?\@!?V?\@:      ?!       B      ?!       J	F????x??F????x??!F????x??R      ?!       Z	F????x??F????x??!F????x??b      ?!       JCPU_ONLYYy/?L????b q?3?*?X@