	o,(ʡa@o,(ʡa@!o,(ʡa@	h|??y???h|??y???!h|??y???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:o,(ʡa@PP?V??0@A?>_@Y??zO????rEagerKernelExecute 0*	?????)?@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2?,^nD@!=gO??W@)?,^nD@1=gO??W@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2????)?E@!?? P?X@)??\???@1?x??a@:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice?s???z??!????????)?s???z??1????????:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip???0?yD@!??!???W@)?:??Tޮ?1??????:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSliceQi??!F?qՐ??)Qi??1F?qՐ??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle0+?~~D@!Cd<?W@)uWv?????15??????:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2sI?v??!?05?1??)sI?v??1?05?1??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch9??U}?!L?<?׻??)9??U}?1L?<?׻??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism"8.????!?A?&???)
ףp=
w?1??
a???:Preprocessing2F
Iterator::Model`??"????!?e?'?ݡ?)??????c?1_(
?D?v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 11.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9i|??y???I?
,Ø?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	PP?V??0@PP?V??0@!PP?V??0@      ?!       "      ?!       *      ?!       2	?>_@?>_@!?>_@:      ?!       B      ?!       J	??zO??????zO????!??zO????R      ?!       Z	??zO??????zO????!??zO????b      ?!       JCPU_ONLYYi|??y???b q?
,Ø?X@