	??Q?Ad@??Q?Ad@!??Q?Ad@	|ŏX???|ŏX???!|ŏX???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??Q?Ad@??S㥛$@A?G?z?b@YˡE?????rEagerKernelExecute 0*	     ??@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2??x?&!D@!?I0?X@)??x?&!D@1?I0?X@:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2㥛? ???!????]???)㥛? ???1????]???:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zipo???AD@!???.<?X@))\???(??1u??Y?O??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice????????!H? 	z??)????????1H? 	z??:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle     @D@!K??X@)?I+???1?}|)???:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice{?G?z??!?m.??){?G?z??1?m.??:Preprocessing2F
Iterator::Model????Mb??!DC{$%??)????Mb??1DC{$%??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9|ŏX???Iw?na?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??S㥛$@??S㥛$@!??S㥛$@      ?!       "      ?!       *      ?!       2	?G?z?b@?G?z?b@!?G?z?b@:      ?!       B      ?!       J	ˡE?????ˡE?????!ˡE?????R      ?!       Z	ˡE?????ˡE?????!ˡE?????b      ?!       JCPU_ONLYY|ŏX???b qw?na?X@