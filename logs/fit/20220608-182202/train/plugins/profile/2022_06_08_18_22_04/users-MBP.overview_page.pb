?	+??X??^@+??X??^@!+??X??^@	
?}????
?}????!
?}????"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:+??X??^@F???!@A)??职\@Y-???b???rEagerKernelExecute 0*	<?O?W??@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2????mF@!? ?Zs?W@)????mF@1? ?Zs?W@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?O?mG@!)\tuv?X@)??mnLO??1???>F?@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip?)s??F@!w??0Q?W@)?p?a?ƣ?1?W??a??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice_{fI????!>t?hu??)_{fI????1>t?hu??:Preprocessing2?
WIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::TensorSlice??a????!`?0M{??)??a????1`?0M{??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle??
?F@!A???W@)n??E????1?ms???:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchzo???!???a51??)zo???1???a51??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???]ڐ?!xZ?}?8??)?@?t??1>????@??:Preprocessing2F
Iterator::Model???(\???!??{?i??)x*???O[?1?q?? ?m?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9
?}????I?EPl??X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	F???!@F???!@!F???!@      ?!       "      ?!       *      ?!       2	)??职\@)??职\@!)??职\@:      ?!       B      ?!       J	-???b???-???b???!-???b???R      ?!       Z	-???b???-???b???!-???b???b      ?!       JCPU_ONLYY
?}????b q?EPl??X@Y      Y@q&kC?G???"?
both?Your program is POTENTIALLY input-bound because 7.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
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