?	?	K<??d@?	K<??d@!?	K<??d@	????8???????8???!????8???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?	K<??d@??[*@A?%?Ѻ?b@Yy????8??rEagerKernelExecute 0*	#??~BM?@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2?e?c]J@!?LhiPW@)?e?c]J@1?LhiPW@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2??	/??K@!c???]?X@)Ǽ?8dC@1??o"F@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip?Udt@J@!a?
[W@)?Բ????1??`????:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice??F!ɬ??!t?M?0n??)??F!ɬ??1t?M?0n??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice3Q??????!iXJfѱ?)3Q??????1iXJfѱ?:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2?a/???!iN?a$??)?a/???1iN?a$??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?????!J@!Jv??]W@)?{???S??1?5ۊ??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?6???N??!?z?I?(??)?6???N??1?z?I?(??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??ᱟŒ?!??;Tɠ?)?fh<??1*lRYhӎ?:Preprocessing2F
Iterator::Model?Ϝ?)ǔ?!i6X3????)??R`?1#?Ճ??l?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????8???I??vL??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??[*@??[*@!??[*@      ?!       "      ?!       *      ?!       2	?%?Ѻ?b@?%?Ѻ?b@!?%?Ѻ?b@:      ?!       B      ?!       J	y????8??y????8??!y????8??R      ?!       Z	y????8??y????8??!y????8??b      ?!       JCPU_ONLYY????8???b q??vL??X@Y      Y@q???7T??"?
both?Your program is POTENTIALLY input-bound because 8.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"CPU: B 