	-?????t@-?????t@!-?????t@	??Q,?????Q,???!??Q,???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:-?????t@/?$??-@AF?????s@Y;?O??n??rEagerKernelExecute 0*	     ?`@2F
Iterator::Modely?&1???!?P^CyE@)y?&1???1?P^CyE@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch?~j?t???!? れB@)?~j?t???1? れB@:Preprocessing2w
@Iterator::Model::MaxIntraOpParallelism::MapAndBatch::TensorSliceV-???!?}s??5@)V-???1?}s??5@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 4.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??Q,???I?:}???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	/?$??-@/?$??-@!/?$??-@      ?!       "      ?!       *      ?!       2	F?????s@F?????s@!F?????s@:      ?!       B      ?!       J	;?O??n??;?O??n??!;?O??n??R      ?!       Z	;?O??n??;?O??n??!;?O??n??b      ?!       JCPU_ONLYY??Q,???b q?:}???X@