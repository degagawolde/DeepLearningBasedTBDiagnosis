	ޭ,????@ޭ,????@!ޭ,????@	??ڤia???ڤia?!??ڤia?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:ޭ,????@?????@A???8+?{@Y??~j?t??rEagerKernelExecute 0*	     ?V@2w
@Iterator::Model::MaxIntraOpParallelism::MapAndBatch::TensorSlice?S㥛İ?!?h?PR@)?S㥛İ?1?h?PR@:Preprocessing2F
Iterator::Model;?O??n??!?W?s??3@);?O??n??1?W?s??3@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatchy?&1?|?!?0&q?@)y?&1?|?1?0&q?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 75.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??ڤia?IK??-??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?????@?????@!?????@      ?!       "      ?!       *      ?!       2	???8+?{@???8+?{@!???8+?{@:      ?!       B      ?!       J	??~j?t????~j?t??!??~j?t??R      ?!       Z	??~j?t????~j?t??!??~j?t??b      ?!       JCPU_ONLYY??ڤia?b qK??-??X@