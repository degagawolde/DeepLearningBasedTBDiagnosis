	???_??]@???_??]@!???_??]@	F?zuvé?F?zuvé?!F?zuvé?"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:???_??]@?6?Ӂ?"@A"? ˂n[@YcD?в??rEagerKernelExecute 0*	MbX??@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV23??p??S@!I!?4lX@)3??p??S@1I!?4lX@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2[[x^*RT@!??犠?X@)??ާ?P??1?*36$+ @:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip?س??S@!<s?>?tX@)h#?M)???1??'???:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice?!U????!Cf???)?!U????1Cf???:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2?#??ŋ??!?q??y??)?#??ŋ??1?q??y??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice+?m????!ʭ5YLG??)+?m????1ʭ5YLG??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle?س??S@!?*6iGwX@)      ??1?P??S???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismj?t???!?^ ????)+l? [v?1h??e?x{?:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch??n?u?!?????z?)??n?u?1?????z?:Preprocessing2F
Iterator::Model??P?n??!&W?l????)x*???O[?1p?????`?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 7.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9G?zuvé?I?P1???X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?6?Ӂ?"@?6?Ӂ?"@!?6?Ӂ?"@      ?!       "      ?!       *      ?!       2	"? ˂n[@"? ˂n[@!"? ˂n[@:      ?!       B      ?!       J	cD?в??cD?в??!cD?в??R      ?!       Z	cD?в??cD?в??!cD?в??b      ?!       JCPU_ONLYYG?zuvé?b q?P1???X@