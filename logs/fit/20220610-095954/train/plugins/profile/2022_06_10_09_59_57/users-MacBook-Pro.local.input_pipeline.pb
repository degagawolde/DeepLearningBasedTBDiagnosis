	?zNz_Oc@?zNz_Oc@!?zNz_Oc@	?+?QS????+?QS???!?+?QS???"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?zNz_Oc@gC??A$@A??
?b@Yn???Wu??rEagerKernelExecute 0*	?G?z???@2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2V??D??G@!+?8H??V@)V??D??G@1+?8H??V@:Preprocessing2p
9Iterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2?????XJ@!@?;?<?X@)$???~K@1t??^S!@:Preprocessing2~
GIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zipp?h??H@!q?????V@)??Ɋ????1??K/???:Preprocessing2y
BIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle????H@!??~??V@)?Z	?%q??1E?QFԼ?:Preprocessing2?
YIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2???j׬?!w)?0@P??)???j׬?1w)?0@P??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice????u???!?ij?jM??)????u???1?ij?jM??:Preprocessing2?
fIterator::Model::MaxIntraOpParallelism::Prefetch::BatchV2::Shuffle::Zip[1]::ParallelMapV2::TensorSlice?EИI??!ݰ? r6??)?EИI??1ݰ? r6??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetchٳ?25	~?!'AZ?q??)ٳ?25	~?1'AZ?q??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism???8???!?(?'6??);?/K;5w?1!??x???:Preprocessing2F
Iterator::Modelp"???ӏ?!旣?I$??)?Ϲ???d?1??	??s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 6.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?+?QS???I??+k?X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	gC??A$@gC??A$@!gC??A$@      ?!       "      ?!       *      ?!       2	??
?b@??
?b@!??
?b@:      ?!       B      ?!       J	n???Wu??n???Wu??!n???Wu??R      ?!       Z	n???Wu??n???Wu??!n???Wu??b      ?!       JCPU_ONLYY?+?QS???b q??+k?X@