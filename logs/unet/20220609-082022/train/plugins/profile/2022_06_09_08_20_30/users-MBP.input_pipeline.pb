	???s?@???s?@!???s?@	???Xx????Xx?!???Xx?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???s?@?0
?Ǣw@A?%8??n@Y?l??????rEagerKernelExecute 0*	??v???U@2w
@Iterator::Model::MaxIntraOpParallelism::MapAndBatch::TensorSliceJ?\??!Rd])ibR@)J?\??1Rd])ibR@:Preprocessing2F
Iterator::Model???Q???!?͆B1@)???Q???1?͆B1@:Preprocessing2j
3Iterator::Model::MaxIntraOpParallelism::MapAndBatch????Mb??!?A??h"@)????Mb??1?A??h"@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 60.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???Xx?I0?o???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?0
?Ǣw@?0
?Ǣw@!?0
?Ǣw@      ?!       "      ?!       *      ?!       2	?%8??n@?%8??n@!?%8??n@:      ?!       B      ?!       J	?l???????l??????!?l??????R      ?!       Z	?l???????l??????!?l??????b      ?!       JCPU_ONLYY???Xx?b q0?o???X@