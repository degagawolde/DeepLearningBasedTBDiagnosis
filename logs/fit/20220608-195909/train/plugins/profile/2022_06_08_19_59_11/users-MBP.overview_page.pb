?	?~??^@?~??^@!?~??^@	~#? >f??~#? >f??!~#? >f??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?~??^@w?x??&@A?????[@Y^????rEagerKernelExecute 0*	? ?rȿ?@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV29??!0E@!?9?,??W@)9??!0E@1?9?,??W@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2j?!?>F@!:LM?X@)??,'?4 @1???`?0@:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice??????! ?%????)??????1 ?%????:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zipl??3?7E@!??f?)?W@)?#??t???1HnA?????:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::ShuffleIIC;E@!?A?=?W@)$??;???1.??k+O??:Preprocessing2?
WIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::TensorSlice9??v????!嚓u????)9??v????1嚓u????:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?6???N??!?????˖?)?6???N??1?????˖?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Z'.?+??!f???'??)	?L?nx?1y?p?c??:Preprocessing2F
Iterator::ModelD?l?????!OddUԣ?)?S?K?W?1|N?;??j?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9~#? >f??In???L?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	w?x??&@w?x??&@!w?x??&@      ?!       "      ?!       *      ?!       2	?????[@?????[@!?????[@:      ?!       B      ?!       J	^????^????!^????R      ?!       Z	^????^????!^????b      ?!       JCPU_ONLYY~#? >f??b qn???L?X@Y      Y@q;?|u;??"?
both?Your program is POTENTIALLY input-bound because 9.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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