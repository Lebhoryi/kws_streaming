อก
ัฃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
พ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878๐ค

streaming/stream/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:(*(
shared_namestreaming/stream/states

+streaming/stream/states/Read/ReadVariableOpReadVariableOpstreaming/stream/states*&
_output_shapes
:(*
dtype0

streaming/stream_1/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:&**
shared_namestreaming/stream_1/states

-streaming/stream_1/states/Read/ReadVariableOpReadVariableOpstreaming/stream_1/states*&
_output_shapes
:&*
dtype0

streaming/input_stateVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namestreaming/input_state

)streaming/input_state/Read/ReadVariableOpReadVariableOpstreaming/input_state*
_output_shapes
:	*
dtype0

streaming/stream_2/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namestreaming/stream_2/states

-streaming/stream_2/states/Read/ReadVariableOpReadVariableOpstreaming/stream_2/states*#
_output_shapes
:*
dtype0

streaming/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namestreaming/dense/kernel

*streaming/dense/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense/kernel* 
_output_shapes
:
*
dtype0

streaming/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namestreaming/dense/bias
z
(streaming/dense/bias/Read/ReadVariableOpReadVariableOpstreaming/dense/bias*
_output_shapes	
:*
dtype0

streaming/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_namestreaming/dense_1/kernel

,streaming/dense_1/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense_1/kernel* 
_output_shapes
:
*
dtype0

streaming/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestreaming/dense_1/bias
~
*streaming/dense_1/bias/Read/ReadVariableOpReadVariableOpstreaming/dense_1/bias*
_output_shapes	
:*
dtype0

streaming/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*)
shared_namestreaming/dense_2/kernel

,streaming/dense_2/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense_2/kernel*
_output_shapes
:	*
dtype0

streaming/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestreaming/dense_2/bias
}
*streaming/dense_2/bias/Read/ReadVariableOpReadVariableOpstreaming/dense_2/bias*
_output_shapes
:*
dtype0

stream/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_namestream/conv2d/kernel

(stream/conv2d/kernel/Read/ReadVariableOpReadVariableOpstream/conv2d/kernel*&
_output_shapes
:*
dtype0
|
stream/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_namestream/conv2d/bias
u
&stream/conv2d/bias/Read/ReadVariableOpReadVariableOpstream/conv2d/bias*
_output_shapes
:*
dtype0

stream_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namestream_1/conv2d_1/kernel

,stream_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/kernel*&
_output_shapes
:*
dtype0

stream_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestream_1/conv2d_1/bias
}
*stream_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/bias*
_output_shapes
:*
dtype0

streaming/gru_1/cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ภ*,
shared_namestreaming/gru_1/cell/kernel

/streaming/gru_1/cell/kernel/Read/ReadVariableOpReadVariableOpstreaming/gru_1/cell/kernel* 
_output_shapes
:
ภ*
dtype0
จ
%streaming/gru_1/cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*6
shared_name'%streaming/gru_1/cell/recurrent_kernel
ก
9streaming/gru_1/cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp%streaming/gru_1/cell/recurrent_kernel* 
_output_shapes
:
*
dtype0

streaming/gru_1/cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namestreaming/gru_1/cell/bias

-streaming/gru_1/cell/bias/Read/ReadVariableOpReadVariableOpstreaming/gru_1/cell/bias*
_output_shapes
:	*
dtype0

NoOpNoOp
ภ8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*๛7
value๑7B๎7 B็7
จ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
R
	variables
regularization_losses
trainable_variables
	keras_api
y
cell
state_shape

states
	variables
regularization_losses
trainable_variables
	keras_api
y
cell
state_shape

states
	variables
 regularization_losses
!trainable_variables
"	keras_api
R
#	variables
$regularization_losses
%trainable_variables
&	keras_api
q
'input_state
(gru_cell
)	variables
*regularization_losses
+trainable_variables
,	keras_api
y
-cell
.state_shape

/states
0	variables
1regularization_losses
2trainable_variables
3	keras_api
R
4	variables
5regularization_losses
6trainable_variables
7	keras_api
h

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
h

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
h

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
~
J0
K1
2
L3
M4
5
N6
O7
P8
'9
/10
811
912
>13
?14
D15
E16
 
^
J0
K1
L2
M3
N4
O5
P6
87
98
>9
?10
D11
E12
ญ
Qlayer_metrics

Rlayers
	variables
Smetrics
Tnon_trainable_variables
regularization_losses
Ulayer_regularization_losses
trainable_variables
 
 
 
 
ญ
Vlayer_metrics

Wlayers
	variables
Xmetrics
Ynon_trainable_variables
regularization_losses
Zlayer_regularization_losses
trainable_variables
h

Jkernel
Kbias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
 
ca
VARIABLE_VALUEstreaming/stream/states6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUE

J0
K1
2
 

J0
K1
ญ
_layer_metrics

`layers
	variables
ametrics
bnon_trainable_variables
regularization_losses
clayer_regularization_losses
trainable_variables
h

Lkernel
Mbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_1/states6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUE

L0
M1
2
 

L0
M1
ญ
hlayer_metrics

ilayers
	variables
jmetrics
knon_trainable_variables
 regularization_losses
llayer_regularization_losses
!trainable_variables
 
 
 
ญ
mlayer_metrics

nlayers
#	variables
ometrics
pnon_trainable_variables
$regularization_losses
qlayer_regularization_losses
%trainable_variables
fd
VARIABLE_VALUEstreaming/input_state;layer_with_weights-2/input_state/.ATTRIBUTES/VARIABLE_VALUE
~

Nkernel
Orecurrent_kernel
Pbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api

N0
O1
P2
'3
 

N0
O1
P2
ญ
vlayer_metrics

wlayers
)	variables
xmetrics
ynon_trainable_variables
*regularization_losses
zlayer_regularization_losses
+trainable_variables
R
{	variables
|regularization_losses
}trainable_variables
~	keras_api
 
ec
VARIABLE_VALUEstreaming/stream_2/states6layer_with_weights-3/states/.ATTRIBUTES/VARIABLE_VALUE

/0
 
 
ฑ
layer_metrics
layers
0	variables
metrics
non_trainable_variables
1regularization_losses
 layer_regularization_losses
2trainable_variables
 
 
 
ฒ
layer_metrics
layers
4	variables
metrics
non_trainable_variables
5regularization_losses
 layer_regularization_losses
6trainable_variables
b`
VARIABLE_VALUEstreaming/dense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEstreaming/dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

80
91
 

80
91
ฒ
layer_metrics
layers
:	variables
metrics
non_trainable_variables
;regularization_losses
 layer_regularization_losses
<trainable_variables
db
VARIABLE_VALUEstreaming/dense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstreaming/dense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

>0
?1
 

>0
?1
ฒ
layer_metrics
layers
@	variables
metrics
non_trainable_variables
Aregularization_losses
 layer_regularization_losses
Btrainable_variables
db
VARIABLE_VALUEstreaming/dense_2/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstreaming/dense_2/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 

D0
E1
ฒ
layer_metrics
layers
F	variables
metrics
non_trainable_variables
Gregularization_losses
 layer_regularization_losses
Htrainable_variables
PN
VARIABLE_VALUEstream/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEstream/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_1/conv2d_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_1/conv2d_1/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEstreaming/gru_1/cell/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUE%streaming/gru_1/cell/recurrent_kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEstreaming/gru_1/cell/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
 
N
0
1
2
3
4
5
6
7
	8

9
10
 

0
1
'2
/3
 
 
 
 
 
 

J0
K1
 

J0
K1
ฒ
layer_metrics
layers
[	variables
metrics
non_trainable_variables
\regularization_losses
 layer_regularization_losses
]trainable_variables
 

0
 

0
 

L0
M1
 

L0
M1
ฒ
layer_metrics
layers
d	variables
metrics
?non_trainable_variables
eregularization_losses
 กlayer_regularization_losses
ftrainable_variables
 

0
 

0
 
 
 
 
 
 

N0
O1
P2
 

N0
O1
P2
ฒ
ขlayer_metrics
ฃlayers
r	variables
คmetrics
ฅnon_trainable_variables
sregularization_losses
 ฆlayer_regularization_losses
ttrainable_variables
 

(0
 

'0
 
 
 
 
ฒ
งlayer_metrics
จlayers
{	variables
ฉmetrics
ชnon_trainable_variables
|regularization_losses
 ซlayer_regularization_losses
}trainable_variables
 

-0
 

/0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
t
serving_default_input_audioPlaceholder*"
_output_shapes
:(*
dtype0*
shape:(
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_audiostreaming/stream/statesstream/conv2d/kernelstream/conv2d/biasstreaming/stream_1/statesstream_1/conv2d_1/kernelstream_1/conv2d_1/biasstreaming/gru_1/cell/biasstreaming/gru_1/cell/kernelstreaming/input_state%streaming/gru_1/cell/recurrent_kernelstreaming/stream_2/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstreaming/dense_2/kernelstreaming/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_5783
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ฑ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+streaming/stream/states/Read/ReadVariableOp-streaming/stream_1/states/Read/ReadVariableOp)streaming/input_state/Read/ReadVariableOp-streaming/stream_2/states/Read/ReadVariableOp*streaming/dense/kernel/Read/ReadVariableOp(streaming/dense/bias/Read/ReadVariableOp,streaming/dense_1/kernel/Read/ReadVariableOp*streaming/dense_1/bias/Read/ReadVariableOp,streaming/dense_2/kernel/Read/ReadVariableOp*streaming/dense_2/bias/Read/ReadVariableOp(stream/conv2d/kernel/Read/ReadVariableOp&stream/conv2d/bias/Read/ReadVariableOp,stream_1/conv2d_1/kernel/Read/ReadVariableOp*stream_1/conv2d_1/bias/Read/ReadVariableOp/streaming/gru_1/cell/kernel/Read/ReadVariableOp9streaming/gru_1/cell/recurrent_kernel/Read/ReadVariableOp-streaming/gru_1/cell/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_6707
ุ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestreaming/stream/statesstreaming/stream_1/statesstreaming/input_statestreaming/stream_2/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstreaming/dense_2/kernelstreaming/dense_2/biasstream/conv2d/kernelstream/conv2d/biasstream_1/conv2d_1/kernelstream_1/conv2d_1/biasstreaming/gru_1/cell/kernel%streaming/gru_1/cell/recurrent_kernelstreaming/gru_1/cell/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_6768ฑ

ข
B__inference_stream_2_layer_call_and_return_conditional_losses_5513

inputs,
(readvariableop_streaming_stream_2_states
identityขAssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*#
_output_shapes
:2
concatง
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_2_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshapew
IdentityIdentityflatten/Reshape:output:0^AssignVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:
 
_user_specified_nameinputs
๔
ื
"__inference_signature_wrapper_5783
input_audio
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_2_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_2_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_52812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
"
_output_shapes
:(
%
_user_specified_nameinput_audio
ด
ว
A__inference_dense_1_layer_call_and_return_conditional_losses_5586

inputs2
.matmul_readvariableop_streaming_dense_1_kernel1
-biasadd_readvariableop_streaming_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAddP
ReluReluBiasAdd:output:0*
T0*
_output_shapes
:	2
Relu^
IdentityIdentityRelu:activations:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
็

$__inference_dense_layer_call_fn_6598

inputs
streaming_dense_kernel
streaming_dense_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_kernelstreaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_55632
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
ญ
ฬ
__inference__wrapped_model_5281
input_audio>
:functional_1_stream_readvariableop_streaming_stream_statesI
Efunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernelH
Dfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_biasB
>functional_1_stream_1_readvariableop_streaming_stream_1_statesQ
Mfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelP
Lfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasD
@functional_1_gru_1_cell_readvariableop_streaming_gru_1_cell_biasM
Ifunctional_1_gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernelI
Efunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state[
Wfunctional_1_gru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernelB
>functional_1_stream_2_readvariableop_streaming_stream_2_statesC
?functional_1_dense_matmul_readvariableop_streaming_dense_kernelB
>functional_1_dense_biasadd_readvariableop_streaming_dense_biasG
Cfunctional_1_dense_1_matmul_readvariableop_streaming_dense_1_kernelF
Bfunctional_1_dense_1_biasadd_readvariableop_streaming_dense_1_biasG
Cfunctional_1_dense_2_matmul_readvariableop_streaming_dense_2_kernelF
Bfunctional_1_dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityข#functional_1/gru_1/AssignVariableOpข$functional_1/stream/AssignVariableOpข&functional_1/stream_1/AssignVariableOpข&functional_1/stream_2/AssignVariableOpณ
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dim๘
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_audio;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(20
.functional_1/tf_op_layer_ExpandDims/ExpandDimsห
"functional_1/stream/ReadVariableOpReadVariableOp:functional_1_stream_readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02$
"functional_1/stream/ReadVariableOpง
'functional_1/stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2)
'functional_1/stream/strided_slice/stackซ
)functional_1/stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream/strided_slice/stack_1ซ
)functional_1/stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)functional_1/stream/strided_slice/stack_2?
!functional_1/stream/strided_sliceStridedSlice*functional_1/stream/ReadVariableOp:value:00functional_1/stream/strided_slice/stack:output:02functional_1/stream/strided_slice/stack_1:output:02functional_1/stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2#
!functional_1/stream/strided_slice
functional_1/stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/stream/concat/axis
functional_1/stream/concatConcatV2*functional_1/stream/strided_slice:output:07functional_1/tf_op_layer_ExpandDims/ExpandDims:output:0(functional_1/stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:(2
functional_1/stream/concat
$functional_1/stream/AssignVariableOpAssignVariableOp:functional_1_stream_readvariableop_streaming_stream_states#functional_1/stream/concat:output:0#^functional_1/stream/ReadVariableOp*
_output_shapes
 *
dtype02&
$functional_1/stream/AssignVariableOp
0functional_1/stream/conv2d/Conv2D/ReadVariableOpReadVariableOpEfunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel%^functional_1/stream/AssignVariableOp*&
_output_shapes
:*
dtype022
0functional_1/stream/conv2d/Conv2D/ReadVariableOp
!functional_1/stream/conv2d/Conv2DConv2D#functional_1/stream/concat:output:08functional_1/stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2#
!functional_1/stream/conv2d/Conv2D
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpDfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_bias%^functional_1/stream/AssignVariableOp*
_output_shapes
:*
dtype023
1functional_1/stream/conv2d/BiasAdd/ReadVariableOp๋
"functional_1/stream/conv2d/BiasAddBiasAdd*functional_1/stream/conv2d/Conv2D:output:09functional_1/stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2$
"functional_1/stream/conv2d/BiasAddจ
functional_1/stream/conv2d/ReluRelu+functional_1/stream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2!
functional_1/stream/conv2d/Reluำ
$functional_1/stream_1/ReadVariableOpReadVariableOp>functional_1_stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02&
$functional_1/stream_1/ReadVariableOpซ
)functional_1/stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_1/strided_slice/stackฏ
+functional_1/stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_1/strided_slice/stack_1ฏ
+functional_1/stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_1/strided_slice/stack_2
#functional_1/stream_1/strided_sliceStridedSlice,functional_1/stream_1/ReadVariableOp:value:02functional_1/stream_1/strided_slice/stack:output:04functional_1/stream_1/strided_slice/stack_1:output:04functional_1/stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2%
#functional_1/stream_1/strided_slice
!functional_1/stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_1/concat/axis
functional_1/stream_1/concatConcatV2,functional_1/stream_1/strided_slice:output:0-functional_1/stream/conv2d/Relu:activations:0*functional_1/stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:&2
functional_1/stream_1/concat
&functional_1/stream_1/AssignVariableOpAssignVariableOp>functional_1_stream_1_readvariableop_streaming_stream_1_states%functional_1/stream_1/concat:output:0%^functional_1/stream_1/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_1/AssignVariableOpซ
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel'^functional_1/stream_1/AssignVariableOp*&
_output_shapes
:*
dtype026
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp
%functional_1/stream_1/conv2d_1/Conv2DConv2D%functional_1/stream_1/concat:output:0<functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2'
%functional_1/stream_1/conv2d_1/Conv2D?
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias'^functional_1/stream_1/AssignVariableOp*
_output_shapes
:*
dtype027
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOp๛
&functional_1/stream_1/conv2d_1/BiasAddBiasAdd.functional_1/stream_1/conv2d_1/Conv2D:output:0=functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2(
&functional_1/stream_1/conv2d_1/BiasAddด
#functional_1/stream_1/conv2d_1/ReluRelu/functional_1/stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2%
#functional_1/stream_1/conv2d_1/Relu
functional_1/reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
functional_1/reshape/Shape
(functional_1/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(functional_1/reshape/strided_slice/stackข
*functional_1/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_1/reshape/strided_slice/stack_1ข
*functional_1/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*functional_1/reshape/strided_slice/stack_2เ
"functional_1/reshape/strided_sliceStridedSlice#functional_1/reshape/Shape:output:01functional_1/reshape/strided_slice/stack:output:03functional_1/reshape/strided_slice/stack_1:output:03functional_1/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"functional_1/reshape/strided_slice
$functional_1/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2&
$functional_1/reshape/Reshape/shape/1
$functional_1/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2&
$functional_1/reshape/Reshape/shape/2
"functional_1/reshape/Reshape/shapePack+functional_1/reshape/strided_slice:output:0-functional_1/reshape/Reshape/shape/1:output:0-functional_1/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2$
"functional_1/reshape/Reshape/shapeี
functional_1/reshape/ReshapeReshape1functional_1/stream_1/conv2d_1/Relu:activations:0+functional_1/reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:ภ2
functional_1/reshape/Reshapeซ
functional_1/gru_1/SqueezeSqueeze%functional_1/reshape/Reshape:output:0*
T0*
_output_shapes
:	ภ*
squeeze_dims
2
functional_1/gru_1/Squeezeา
&functional_1/gru_1/cell/ReadVariableOpReadVariableOp@functional_1_gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02(
&functional_1/gru_1/cell/ReadVariableOpด
functional_1/gru_1/cell/unstackUnpack.functional_1/gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2!
functional_1/gru_1/cell/unstack๊
-functional_1/gru_1/cell/MatMul/ReadVariableOpReadVariableOpIfunctional_1_gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02/
-functional_1/gru_1/cell/MatMul/ReadVariableOpะ
functional_1/gru_1/cell/MatMulMatMul#functional_1/gru_1/Squeeze:output:05functional_1/gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2 
functional_1/gru_1/cell/MatMulห
functional_1/gru_1/cell/BiasAddBiasAdd(functional_1/gru_1/cell/MatMul:product:0(functional_1/gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	2!
functional_1/gru_1/cell/BiasAdd
functional_1/gru_1/cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
functional_1/gru_1/cell/Const
'functional_1/gru_1/cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2)
'functional_1/gru_1/cell/split/split_dim๔
functional_1/gru_1/cell/splitSplit0functional_1/gru_1/cell/split/split_dim:output:0(functional_1/gru_1/cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
functional_1/gru_1/cell/split้
/functional_1/gru_1/cell/MatMul_1/ReadVariableOpReadVariableOpEfunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype021
/functional_1/gru_1/cell/MatMul_1/ReadVariableOp
1functional_1/gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpWfunctional_1_gru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype023
1functional_1/gru_1/cell/MatMul_1/ReadVariableOp_1์
 functional_1/gru_1/cell/MatMul_1MatMul7functional_1/gru_1/cell/MatMul_1/ReadVariableOp:value:09functional_1/gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2"
 functional_1/gru_1/cell/MatMul_1ั
!functional_1/gru_1/cell/BiasAdd_1BiasAdd*functional_1/gru_1/cell/MatMul_1:product:0(functional_1/gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	2#
!functional_1/gru_1/cell/BiasAdd_1
functional_1/gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2!
functional_1/gru_1/cell/Const_1ก
)functional_1/gru_1/cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2+
)functional_1/gru_1/cell/split_1/split_dimณ
functional_1/gru_1/cell/split_1SplitV*functional_1/gru_1/cell/BiasAdd_1:output:0(functional_1/gru_1/cell/Const_1:output:02functional_1/gru_1/cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
functional_1/gru_1/cell/split_1ฟ
functional_1/gru_1/cell/addAddV2&functional_1/gru_1/cell/split:output:0(functional_1/gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add
functional_1/gru_1/cell/SigmoidSigmoidfunctional_1/gru_1/cell/add:z:0*
T0*
_output_shapes
:	2!
functional_1/gru_1/cell/Sigmoidร
functional_1/gru_1/cell/add_1AddV2&functional_1/gru_1/cell/split:output:1(functional_1/gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add_1
!functional_1/gru_1/cell/Sigmoid_1Sigmoid!functional_1/gru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2#
!functional_1/gru_1/cell/Sigmoid_1ผ
functional_1/gru_1/cell/mulMul%functional_1/gru_1/cell/Sigmoid_1:y:0(functional_1/gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/mulบ
functional_1/gru_1/cell/add_2AddV2&functional_1/gru_1/cell/split:output:2functional_1/gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add_2
functional_1/gru_1/cell/TanhTanh!functional_1/gru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/Tanhใ
,functional_1/gru_1/cell/mul_1/ReadVariableOpReadVariableOpEfunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02.
,functional_1/gru_1/cell/mul_1/ReadVariableOpส
functional_1/gru_1/cell/mul_1Mul#functional_1/gru_1/cell/Sigmoid:y:04functional_1/gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/mul_1
functional_1/gru_1/cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
functional_1/gru_1/cell/sub/xธ
functional_1/gru_1/cell/subSub&functional_1/gru_1/cell/sub/x:output:0#functional_1/gru_1/cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/subฒ
functional_1/gru_1/cell/mul_2Mulfunctional_1/gru_1/cell/sub:z:0 functional_1/gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/mul_2ท
functional_1/gru_1/cell/add_3AddV2!functional_1/gru_1/cell/mul_1:z:0!functional_1/gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add_3ฬ
#functional_1/gru_1/AssignVariableOpAssignVariableOpEfunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state!functional_1/gru_1/cell/add_3:z:00^functional_1/gru_1/cell/MatMul_1/ReadVariableOp-^functional_1/gru_1/cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02%
#functional_1/gru_1/AssignVariableOpฎ
!functional_1/gru_1/ExpandDims/dimConst$^functional_1/gru_1/AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/gru_1/ExpandDims/dimษ
functional_1/gru_1/ExpandDims
ExpandDims!functional_1/gru_1/cell/add_3:z:0*functional_1/gru_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
functional_1/gru_1/ExpandDimsะ
$functional_1/stream_2/ReadVariableOpReadVariableOp>functional_1_stream_2_readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02&
$functional_1/stream_2/ReadVariableOpซ
)functional_1/stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2+
)functional_1/stream_2/strided_slice/stackฏ
+functional_1/stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2-
+functional_1/stream_2/strided_slice/stack_1ฏ
+functional_1/stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2-
+functional_1/stream_2/strided_slice/stack_2
#functional_1/stream_2/strided_sliceStridedSlice,functional_1/stream_2/ReadVariableOp:value:02functional_1/stream_2/strided_slice/stack:output:04functional_1/stream_2/strided_slice/stack_1:output:04functional_1/stream_2/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2%
#functional_1/stream_2/strided_slice
!functional_1/stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/stream_2/concat/axis
functional_1/stream_2/concatConcatV2,functional_1/stream_2/strided_slice:output:0&functional_1/gru_1/ExpandDims:output:0*functional_1/stream_2/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
functional_1/stream_2/concat
&functional_1/stream_2/AssignVariableOpAssignVariableOp>functional_1_stream_2_readvariableop_streaming_stream_2_states%functional_1/stream_2/concat:output:0%^functional_1/stream_2/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/stream_2/AssignVariableOpฤ
#functional_1/stream_2/flatten/ConstConst'^functional_1/stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2%
#functional_1/stream_2/flatten/Constุ
%functional_1/stream_2/flatten/ReshapeReshape%functional_1/stream_2/concat:output:0,functional_1/stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2'
%functional_1/stream_2/flatten/Reshapeค
functional_1/dropout/IdentityIdentity.functional_1/stream_2/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/Identityึ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp?functional_1_dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpฤ
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/MatMulา
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp>functional_1_dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpล
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/BiasAdd?
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOpCfunctional_1_dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpว
functional_1/dense_1/MatMulMatMul#functional_1/dense/BiasAdd:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/MatMulฺ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpอ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/dense_1/Relu?
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOpCfunctional_1_dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpส
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/MatMulู
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOpBfunctional_1_dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpฬ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/BiasAdd
IdentityIdentity%functional_1/dense_2/BiasAdd:output:0$^functional_1/gru_1/AssignVariableOp%^functional_1/stream/AssignVariableOp'^functional_1/stream_1/AssignVariableOp'^functional_1/stream_2/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::2J
#functional_1/gru_1/AssignVariableOp#functional_1/gru_1/AssignVariableOp2L
$functional_1/stream/AssignVariableOp$functional_1/stream/AssignVariableOp2P
&functional_1/stream_1/AssignVariableOp&functional_1/stream_1/AssignVariableOp2P
&functional_1/stream_2/AssignVariableOp&functional_1/stream_2/AssignVariableOp:X T
+
_output_shapes
:?????????(
%
_user_specified_nameinput_audio
ง
?
+__inference_functional_1_layer_call_fn_6042

inputs
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_2_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCallฆ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_2_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_56862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
ธ
ญ
%__inference_stream_layer_call_fn_6383

inputs
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_conv2d_kernelstream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_53182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:&2

Identity"
identityIdentity:output:0*1
_input_shapes 
:(:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:(
 
_user_specified_nameinputs

B
&__inference_reshape_layer_call_fn_6428

inputs
identityป
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ภ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_53782
PartitionedCallh
IdentityIdentityPartitionedCall:output:0*
T0*#
_output_shapes
:ภ2

Identity"
identityIdentity:output:0*%
_input_shapes
:$:N J
&
_output_shapes
:$
 
_user_specified_nameinputs

ข
B__inference_stream_2_layer_call_and_return_conditional_losses_6548

inputs,
(readvariableop_streaming_stream_2_states
identityขAssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2?
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*#
_output_shapes
:2
concatง
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_2_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshapew
IdentityIdentityflatten/Reshape:output:0^AssignVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:
 
_user_specified_nameinputs
ฆ

B__inference_stream_1_layer_call_and_return_conditional_losses_6402

inputs,
(readvariableop_streaming_stream_1_states;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identityขAssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:&2
concatง
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_1_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpำ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^AssignVariableOp*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpฟ
conv2d_1/Conv2DConv2Dconcat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2
conv2d_1/Conv2Dศ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^AssignVariableOp*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpฃ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2
conv2d_1/Relu
IdentityIdentityconv2d_1/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:$2

Identity"
identityIdentity:output:0*1
_input_shapes 
:&:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:&
 
_user_specified_nameinputs
๑

&__inference_dense_2_layer_call_fn_6633

inputs
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_56082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
๓

&__inference_dense_1_layer_call_fn_6616

inputs
streaming_dense_1_kernel
streaming_dense_1_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_dense_1_kernelstreaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_55862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
๕
]
A__inference_reshape_layer_call_and_return_conditional_losses_5378

inputs
identityg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapek
ReshapeReshapeinputsReshape/shape:output:0*
T0*#
_output_shapes
:ภ2	
Reshape`
IdentityIdentityReshape:output:0*
T0*#
_output_shapes
:ภ2

Identity"
identityIdentity:output:0*%
_input_shapes
:$:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
ข/
ฌ
__inference__traced_save_6707
file_prefix6
2savev2_streaming_stream_states_read_readvariableop8
4savev2_streaming_stream_1_states_read_readvariableop4
0savev2_streaming_input_state_read_readvariableop8
4savev2_streaming_stream_2_states_read_readvariableop5
1savev2_streaming_dense_kernel_read_readvariableop3
/savev2_streaming_dense_bias_read_readvariableop7
3savev2_streaming_dense_1_kernel_read_readvariableop5
1savev2_streaming_dense_1_bias_read_readvariableop7
3savev2_streaming_dense_2_kernel_read_readvariableop5
1savev2_streaming_dense_2_bias_read_readvariableop3
/savev2_stream_conv2d_kernel_read_readvariableop1
-savev2_stream_conv2d_bias_read_readvariableop7
3savev2_stream_1_conv2d_1_kernel_read_readvariableop5
1savev2_stream_1_conv2d_1_bias_read_readvariableop:
6savev2_streaming_gru_1_cell_kernel_read_readvariableopD
@savev2_streaming_gru_1_cell_recurrent_kernel_read_readvariableop8
4savev2_streaming_gru_1_cell_bias_read_readvariableop
savev2_const

identity_1ขMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_12c4da346d9a47639926629217183dcb/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardฆ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename่
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*๚
value๐BํB6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/input_state/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesฌ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesษ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_streaming_stream_states_read_readvariableop4savev2_streaming_stream_1_states_read_readvariableop0savev2_streaming_input_state_read_readvariableop4savev2_streaming_stream_2_states_read_readvariableop1savev2_streaming_dense_kernel_read_readvariableop/savev2_streaming_dense_bias_read_readvariableop3savev2_streaming_dense_1_kernel_read_readvariableop1savev2_streaming_dense_1_bias_read_readvariableop3savev2_streaming_dense_2_kernel_read_readvariableop1savev2_streaming_dense_2_bias_read_readvariableop/savev2_stream_conv2d_kernel_read_readvariableop-savev2_stream_conv2d_bias_read_readvariableop3savev2_stream_1_conv2d_1_kernel_read_readvariableop1savev2_stream_1_conv2d_1_bias_read_readvariableop6savev2_streaming_gru_1_cell_kernel_read_readvariableop@savev2_streaming_gru_1_cell_recurrent_kernel_read_readvariableop4savev2_streaming_gru_1_cell_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
22
SaveV2บ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesก
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*แ
_input_shapesฯ
ฬ: :(:&:	::
::
::	::::::
ภ:
:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:(:,(
&
_output_shapes
:&:%!

_output_shapes
:	:)%
#
_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%	!

_output_shapes
:	: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::&"
 
_output_shapes
:
ภ:&"
 
_output_shapes
:
:%!

_output_shapes
:	:

_output_shapes
: 
ใ&
โ
?__inference_gru_1_layer_call_and_return_conditional_losses_6516

inputs1
-cell_readvariableop_streaming_gru_1_cell_bias:
6cell_matmul_readvariableop_streaming_gru_1_cell_kernel6
2cell_matmul_1_readvariableop_streaming_input_stateH
Dcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel
identityขAssignVariableOpf
SqueezeSqueezeinputs*
T0*
_output_shapes
:	ภ*
squeeze_dims
2	
Squeeze
cell/ReadVariableOpReadVariableOp-cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
cell/ReadVariableOp{
cell/unstackUnpackcell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/unstackฑ
cell/MatMul/ReadVariableOpReadVariableOp6cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02
cell/MatMul/ReadVariableOp
cell/MatMulMatMulSqueeze:output:0"cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/MatMul
cell/BiasAddBiasAddcell/MatMul:product:0cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/BiasAddZ

cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

cell/Constw
cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/split/split_dimจ

cell/splitSplitcell/split/split_dim:output:0cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2

cell/splitฐ
cell/MatMul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/MatMul_1/ReadVariableOpว
cell/MatMul_1/ReadVariableOp_1ReadVariableOpDcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02 
cell/MatMul_1/ReadVariableOp_1?
cell/MatMul_1MatMul$cell/MatMul_1/ReadVariableOp:value:0&cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
cell/MatMul_1
cell/BiasAdd_1BiasAddcell/MatMul_1:product:0cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/BiasAdd_1q
cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/Const_1{
cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/split_1/split_dimิ
cell/split_1SplitVcell/BiasAdd_1:output:0cell/Const_1:output:0cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/split_1s
cell/addAddV2cell/split:output:0cell/split_1:output:0*
T0*
_output_shapes
:	2

cell/add_
cell/SigmoidSigmoidcell/add:z:0*
T0*
_output_shapes
:	2
cell/Sigmoidw

cell/add_1AddV2cell/split:output:1cell/split_1:output:1*
T0*
_output_shapes
:	2

cell/add_1e
cell/Sigmoid_1Sigmoidcell/add_1:z:0*
T0*
_output_shapes
:	2
cell/Sigmoid_1p
cell/mulMulcell/Sigmoid_1:y:0cell/split_1:output:2*
T0*
_output_shapes
:	2

cell/muln

cell/add_2AddV2cell/split:output:2cell/mul:z:0*
T0*
_output_shapes
:	2

cell/add_2X
	cell/TanhTanhcell/add_2:z:0*
T0*
_output_shapes
:	2
	cell/Tanhช
cell/mul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/mul_1/ReadVariableOp~

cell/mul_1Mulcell/Sigmoid:y:0!cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

cell/mul_1]

cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cell/sub/xl
cell/subSubcell/sub/x:output:0cell/Sigmoid:y:0*
T0*
_output_shapes
:	2

cell/subf

cell/mul_2Mulcell/sub:z:0cell/Tanh:y:0*
T0*
_output_shapes
:	2

cell/mul_2k

cell/add_3AddV2cell/mul_1:z:0cell/mul_2:z:0*
T0*
_output_shapes
:	2

cell/add_3ฺ
AssignVariableOpAssignVariableOp2cell_matmul_1_readvariableop_streaming_input_statecell/add_3:z:0^cell/MatMul_1/ReadVariableOp^cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpu
ExpandDims/dimConst^AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim}

ExpandDims
ExpandDimscell/add_3:z:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDimsv
IdentityIdentityExpandDims:output:0^AssignVariableOp*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ภ::::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:ภ
 
_user_specified_nameinputs
ฆ

B__inference_stream_1_layer_call_and_return_conditional_losses_5351

inputs,
(readvariableop_streaming_stream_1_states;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identityขAssignVariableOp
ReadVariableOpReadVariableOp(readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:&2
concatง
AssignVariableOpAssignVariableOp(readvariableop_streaming_stream_1_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpำ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^AssignVariableOp*&
_output_shapes
:*
dtype02 
conv2d_1/Conv2D/ReadVariableOpฟ
conv2d_1/Conv2DConv2Dconcat:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2
conv2d_1/Conv2Dศ
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^AssignVariableOp*
_output_shapes
:*
dtype02!
conv2d_1/BiasAdd/ReadVariableOpฃ
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2
conv2d_1/Relu
IdentityIdentityconv2d_1/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:$2

Identity"
identityIdentity:output:0*1
_input_shapes 
:&:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:&
 
_user_specified_nameinputs
ุ

`
A__inference_dropout_layer_call_and_return_conditional_losses_5535

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ไ8?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shapeฌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬฬ=2
dropout/GreaterEqual/yถ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ฝ
?	
F__inference_functional_1_layer_call_and_return_conditional_losses_5905

inputs1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel5
1stream_2_readvariableop_streaming_stream_2_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpขstream_1/AssignVariableOpขstream_2/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimฬ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(2#
!tf_op_layer_ExpandDims/ExpandDimsค
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฎ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisฬ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:(2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpๅ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpี
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dฺ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOpท
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2
stream/conv2d/Reluฌ
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2บ
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisส
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:&2
stream_1/concatิ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp๗
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpใ
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2D์
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpว
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2
reshape/Reshape/shape/2ศ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeก
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:ภ2
reshape/Reshape
gru_1/SqueezeSqueezereshape/Reshape:output:0*
T0*
_output_shapes
:	ภ*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_1/cell/unstackร
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	2
gru_1/cell/BiasAddf
gru_1/cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/cell/Const
gru_1/cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split/split_dimภ
gru_1/cell/splitSplit#gru_1/cell/split/split_dim:output:0gru_1/cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_1/cell/Const_1
gru_1/cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split_1/split_dim๒
gru_1/cell/split_1SplitVgru_1/cell/BiasAdd_1:output:0gru_1/cell/Const_1:output:0%gru_1/cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_1i
gru_1/cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/cell/sub/x
gru_1/cell/subSubgru_1/cell/sub/x:output:0gru_1/cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_3?
gru_1/AssignVariableOpAssignVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_stategru_1/cell/add_3:z:0#^gru_1/cell/MatMul_1/ReadVariableOp ^gru_1/cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru_1/AssignVariableOp
gru_1/ExpandDims/dimConst^gru_1/AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
gru_1/ExpandDims/dim
gru_1/ExpandDims
ExpandDimsgru_1/cell/add_3:z:0gru_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru_1/ExpandDimsฉ
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2ต
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisภ
stream_2/concatConcatV2stream_2/strided_slice:output:0gru_1/ExpandDims:output:0stream_2/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream_2/concatิ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp
stream_2/flatten/ConstConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Constค
stream_2/flatten/ReshapeReshapestream_2/concat:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ไ8?2
dropout/dropout/Const
dropout/dropout/MulMul!stream_2/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/dropout/Shapeฤ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬฬ=2 
dropout/dropout/GreaterEqual/yึ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1ฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMulซ
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAddท
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMulณ
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Reluถ
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMulฒ
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddฮ
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
๕
]
A__inference_reshape_layer_call_and_return_conditional_losses_6423

inputs
identityg
ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2โ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicem
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2
Reshape/shape/2?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapek
ReshapeReshapeinputsReshape/shape:output:0*
T0*#
_output_shapes
:ภ2	
Reshape`
IdentityIdentityReshape:output:0*
T0*#
_output_shapes
:ภ2

Identity"
identityIdentity:output:0*%
_input_shapes
:$:N J
&
_output_shapes
:$
 
_user_specified_nameinputs
ๅ
โ	
F__inference_functional_1_layer_call_and_return_conditional_losses_6301
input_audio1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel5
1stream_2_readvariableop_streaming_stream_2_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpขstream_1/AssignVariableOpขstream_2/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimั
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_audio.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(2#
!tf_op_layer_ExpandDims/ExpandDimsค
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฎ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisฬ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:(2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpๅ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpี
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dฺ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOpท
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2
stream/conv2d/Reluฌ
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2บ
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisส
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:&2
stream_1/concatิ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp๗
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpใ
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2D์
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpว
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2
reshape/Reshape/shape/2ศ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeก
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:ภ2
reshape/Reshape
gru_1/SqueezeSqueezereshape/Reshape:output:0*
T0*
_output_shapes
:	ภ*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_1/cell/unstackร
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	2
gru_1/cell/BiasAddf
gru_1/cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/cell/Const
gru_1/cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split/split_dimภ
gru_1/cell/splitSplit#gru_1/cell/split/split_dim:output:0gru_1/cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_1/cell/Const_1
gru_1/cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split_1/split_dim๒
gru_1/cell/split_1SplitVgru_1/cell/BiasAdd_1:output:0gru_1/cell/Const_1:output:0%gru_1/cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_1i
gru_1/cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/cell/sub/x
gru_1/cell/subSubgru_1/cell/sub/x:output:0gru_1/cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_3?
gru_1/AssignVariableOpAssignVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_stategru_1/cell/add_3:z:0#^gru_1/cell/MatMul_1/ReadVariableOp ^gru_1/cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru_1/AssignVariableOp
gru_1/ExpandDims/dimConst^gru_1/AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
gru_1/ExpandDims/dim
gru_1/ExpandDims
ExpandDimsgru_1/cell/add_3:z:0gru_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru_1/ExpandDimsฉ
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2ต
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisภ
stream_2/concatConcatV2stream_2/strided_slice:output:0gru_1/ExpandDims:output:0stream_2/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream_2/concatิ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp
stream_2/flatten/ConstConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Constค
stream_2/flatten/ReshapeReshapestream_2/concat:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshape}
dropout/IdentityIdentity!stream_2/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identityฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMulซ
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAddท
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMulณ
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Reluถ
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMulฒ
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddฮ
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp:X T
+
_output_shapes
:?????????(
%
_user_specified_nameinput_audio
อ3
ๆ
F__inference_functional_1_layer_call_and_return_conditional_losses_5686

inputs"
stream_streaming_stream_states
stream_stream_conv2d_kernel
stream_stream_conv2d_bias&
"stream_1_streaming_stream_1_states%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias#
gru_1_streaming_gru_1_cell_bias%
!gru_1_streaming_gru_1_cell_kernel
gru_1_streaming_input_state/
+gru_1_streaming_gru_1_cell_recurrent_kernel&
"stream_2_streaming_stream_2_states 
dense_streaming_dense_kernel
dense_streaming_dense_bias$
 dense_1_streaming_dense_1_kernel"
dense_1_streaming_dense_1_bias$
 dense_2_streaming_dense_2_kernel"
dense_2_streaming_dense_2_bias
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdense_2/StatefulPartitionedCallขdropout/StatefulPartitionedCallขgru_1/StatefulPartitionedCallขstream/StatefulPartitionedCallข stream_1/StatefulPartitionedCallข stream_2/StatefulPartitionedCall๛
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_52912(
&tf_op_layer_ExpandDims/PartitionedCall๋
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_53182 
stream/StatefulPartitionedCall๙
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0"stream_1_streaming_stream_1_states!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_53512"
 stream_1/StatefulPartitionedCall๎
reshape/PartitionedCallPartitionedCall)stream_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ภ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_53782
reshape/PartitionedCall
gru_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0gru_1_streaming_gru_1_cell_bias!gru_1_streaming_gru_1_cell_kernelgru_1_streaming_input_state+gru_1_streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_54742
gru_1/StatefulPartitionedCallง
 stream_2/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0"stream_2_streaming_stream_2_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_55132"
 stream_2/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_55352!
dropout/StatefulPartitionedCallป
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_55632
dense/StatefulPartitionedCallว
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_55862!
dense_1/StatefulPartitionedCallศ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 dense_2_streaming_dense_2_kerneldense_2_streaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_56082!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^gru_1/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
ึ
ม
?__inference_dense_layer_call_and_return_conditional_losses_5563

inputs0
,matmul_readvariableop_streaming_dense_kernel/
+biasadd_readvariableop_streaming_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd\
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
ค
_
A__inference_dropout_layer_call_and_return_conditional_losses_5540

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
็

@__inference_stream_layer_call_and_return_conditional_losses_6375

inputs*
&readvariableop_streaming_stream_states5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identityขAssignVariableOp
ReadVariableOpReadVariableOp&readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:(2
concatฅ
AssignVariableOpAssignVariableOp&readvariableop_streaming_stream_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpษ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel^AssignVariableOp*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpน
conv2d/Conv2DConv2Dconcat:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2
conv2d/Conv2Dพ
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias^AssignVariableOp*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2
conv2d/Relu
IdentityIdentityconv2d/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:&2

Identity"
identityIdentity:output:0*1
_input_shapes 
:(:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:(
 
_user_specified_nameinputs
ค
_
A__inference_dropout_layer_call_and_return_conditional_losses_6571

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ฬ
โ	
F__inference_functional_1_layer_call_and_return_conditional_losses_6186
input_audio1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel5
1stream_2_readvariableop_streaming_stream_2_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpขstream_1/AssignVariableOpขstream_2/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimั
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_audio.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(2#
!tf_op_layer_ExpandDims/ExpandDimsค
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฎ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisฬ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:(2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpๅ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpี
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dฺ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOpท
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2
stream/conv2d/Reluฌ
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2บ
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisส
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:&2
stream_1/concatิ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp๗
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpใ
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2D์
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpว
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2
reshape/Reshape/shape/2ศ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeก
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:ภ2
reshape/Reshape
gru_1/SqueezeSqueezereshape/Reshape:output:0*
T0*
_output_shapes
:	ภ*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_1/cell/unstackร
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	2
gru_1/cell/BiasAddf
gru_1/cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/cell/Const
gru_1/cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split/split_dimภ
gru_1/cell/splitSplit#gru_1/cell/split/split_dim:output:0gru_1/cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_1/cell/Const_1
gru_1/cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split_1/split_dim๒
gru_1/cell/split_1SplitVgru_1/cell/BiasAdd_1:output:0gru_1/cell/Const_1:output:0%gru_1/cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_1i
gru_1/cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/cell/sub/x
gru_1/cell/subSubgru_1/cell/sub/x:output:0gru_1/cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_3?
gru_1/AssignVariableOpAssignVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_stategru_1/cell/add_3:z:0#^gru_1/cell/MatMul_1/ReadVariableOp ^gru_1/cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru_1/AssignVariableOp
gru_1/ExpandDims/dimConst^gru_1/AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
gru_1/ExpandDims/dim
gru_1/ExpandDims
ExpandDimsgru_1/cell/add_3:z:0gru_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru_1/ExpandDimsฉ
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2ต
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisภ
stream_2/concatConcatV2stream_2/strided_slice:output:0gru_1/ExpandDims:output:0stream_2/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream_2/concatิ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp
stream_2/flatten/ConstConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Constค
stream_2/flatten/ReshapeReshapestream_2/concat:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ไ8?2
dropout/dropout/Const
dropout/dropout/MulMul!stream_2/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/dropout/Shapeฤ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬฬ=2 
dropout/dropout/GreaterEqual/yึ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1ฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMulซ
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAddท
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMulณ
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Reluถ
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMulฒ
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddฮ
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp:X T
+
_output_shapes
:?????????(
%
_user_specified_nameinput_audio
ฅ2
ฤ
F__inference_functional_1_layer_call_and_return_conditional_losses_5739

inputs"
stream_streaming_stream_states
stream_stream_conv2d_kernel
stream_stream_conv2d_bias&
"stream_1_streaming_stream_1_states%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias#
gru_1_streaming_gru_1_cell_bias%
!gru_1_streaming_gru_1_cell_kernel
gru_1_streaming_input_state/
+gru_1_streaming_gru_1_cell_recurrent_kernel&
"stream_2_streaming_stream_2_states 
dense_streaming_dense_kernel
dense_streaming_dense_bias$
 dense_1_streaming_dense_1_kernel"
dense_1_streaming_dense_1_bias$
 dense_2_streaming_dense_2_kernel"
dense_2_streaming_dense_2_bias
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdense_2/StatefulPartitionedCallขgru_1/StatefulPartitionedCallขstream/StatefulPartitionedCallข stream_1/StatefulPartitionedCallข stream_2/StatefulPartitionedCall๛
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_52912(
&tf_op_layer_ExpandDims/PartitionedCall๋
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_streaming_stream_statesstream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:&*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_53182 
stream/StatefulPartitionedCall๙
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0"stream_1_streaming_stream_1_states!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_53512"
 stream_1/StatefulPartitionedCall๎
reshape/PartitionedCallPartitionedCall)stream_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:ภ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_reshape_layer_call_and_return_conditional_losses_53782
reshape/PartitionedCall
gru_1/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0gru_1_streaming_gru_1_cell_bias!gru_1_streaming_gru_1_cell_kernelgru_1_streaming_input_state+gru_1_streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_54742
gru_1/StatefulPartitionedCallง
 stream_2/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0"stream_2_streaming_stream_2_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_55132"
 stream_2/StatefulPartitionedCall๊
dropout/PartitionedCallPartitionedCall)stream_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_55402
dropout/PartitionedCallณ
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_streaming_dense_kerneldense_streaming_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_55632
dense/StatefulPartitionedCallว
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 dense_1_streaming_dense_1_kerneldense_1_streaming_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_55862!
dense_1/StatefulPartitionedCallศ
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0 dense_2_streaming_dense_2_kerneldense_2_streaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_56082!
dense_2/StatefulPartitionedCall?
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru_1/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
ง
?
+__inference_functional_1_layer_call_fn_6064

inputs
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_2_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCallฆ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_2_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_57392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
ิJ
๐	
 __inference__traced_restore_6768
file_prefix,
(assignvariableop_streaming_stream_states0
,assignvariableop_1_streaming_stream_1_states,
(assignvariableop_2_streaming_input_state0
,assignvariableop_3_streaming_stream_2_states-
)assignvariableop_4_streaming_dense_kernel+
'assignvariableop_5_streaming_dense_bias/
+assignvariableop_6_streaming_dense_1_kernel-
)assignvariableop_7_streaming_dense_1_bias/
+assignvariableop_8_streaming_dense_2_kernel-
)assignvariableop_9_streaming_dense_2_bias,
(assignvariableop_10_stream_conv2d_kernel*
&assignvariableop_11_stream_conv2d_bias0
,assignvariableop_12_stream_1_conv2d_1_kernel.
*assignvariableop_13_stream_1_conv2d_1_bias3
/assignvariableop_14_streaming_gru_1_cell_kernel=
9assignvariableop_15_streaming_gru_1_cell_recurrent_kernel1
-assignvariableop_16_streaming_gru_1_cell_bias
identity_18ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_11ขAssignVariableOp_12ขAssignVariableOp_13ขAssignVariableOp_14ขAssignVariableOp_15ขAssignVariableOp_16ขAssignVariableOp_2ขAssignVariableOp_3ขAssignVariableOp_4ขAssignVariableOp_5ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9๎
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*๚
value๐BํB6layer_with_weights-0/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/input_state/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesฒ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityง
AssignVariableOpAssignVariableOp(assignvariableop_streaming_stream_statesIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ฑ
AssignVariableOp_1AssignVariableOp,assignvariableop_1_streaming_stream_1_statesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ญ
AssignVariableOp_2AssignVariableOp(assignvariableop_2_streaming_input_stateIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ฑ
AssignVariableOp_3AssignVariableOp,assignvariableop_3_streaming_stream_2_statesIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ฎ
AssignVariableOp_4AssignVariableOp)assignvariableop_4_streaming_dense_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ฌ
AssignVariableOp_5AssignVariableOp'assignvariableop_5_streaming_dense_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ฐ
AssignVariableOp_6AssignVariableOp+assignvariableop_6_streaming_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ฎ
AssignVariableOp_7AssignVariableOp)assignvariableop_7_streaming_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ฐ
AssignVariableOp_8AssignVariableOp+assignvariableop_8_streaming_dense_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ฎ
AssignVariableOp_9AssignVariableOp)assignvariableop_9_streaming_dense_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ฐ
AssignVariableOp_10AssignVariableOp(assignvariableop_10_stream_conv2d_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11ฎ
AssignVariableOp_11AssignVariableOp&assignvariableop_11_stream_conv2d_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ด
AssignVariableOp_12AssignVariableOp,assignvariableop_12_stream_1_conv2d_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13ฒ
AssignVariableOp_13AssignVariableOp*assignvariableop_13_stream_1_conv2d_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ท
AssignVariableOp_14AssignVariableOp/assignvariableop_14_streaming_gru_1_cell_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ม
AssignVariableOp_15AssignVariableOp9assignvariableop_15_streaming_gru_1_cell_recurrent_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ต
AssignVariableOp_16AssignVariableOp-assignvariableop_16_streaming_gru_1_cell_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpิ
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17ว
Identity_18IdentityIdentity_17:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_18"#
identity_18Identity_18:output:0*Y
_input_shapesH
F: :::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_5291

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:(2

Identity"
identityIdentity:output:0*!
_input_shapes
:(:J F
"
_output_shapes
:(
 
_user_specified_nameinputs
๘
_
&__inference_dropout_layer_call_fn_6576

inputs
identityขStatefulPartitionedCallฯ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_55352
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
ุ

`
A__inference_dropout_layer_call_and_return_conditional_losses_6566

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ไ8?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shapeฌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *อฬฬ=2
dropout/GreaterEqual/yถ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs

ใ
$__inference_gru_1_layer_call_fn_6534

inputs
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
identityขStatefulPartitionedCallะ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_54742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ภ::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ภ
 
_user_specified_nameinputs
ด
ว
A__inference_dense_1_layer_call_and_return_conditional_losses_6609

inputs2
.matmul_readvariableop_streaming_dense_1_kernel1
-biasadd_readvariableop_streaming_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAddP
ReluReluBiasAdd:output:0*
T0*
_output_shapes
:	2
Relu^
IdentityIdentityRelu:activations:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
ถ
เ
+__inference_functional_1_layer_call_fn_6345
input_audio
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_2_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCallซ
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_2_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_57392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????(
%
_user_specified_nameinput_audio

ใ
$__inference_gru_1_layer_call_fn_6525

inputs
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
identityขStatefulPartitionedCallะ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_54742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ภ::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:ภ
 
_user_specified_nameinputs
?
ว
A__inference_dense_2_layer_call_and_return_conditional_losses_6626

inputs2
.matmul_readvariableop_streaming_dense_2_kernel1
-biasadd_readvariableop_streaming_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAdd[
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
ึ
?	
F__inference_functional_1_layer_call_and_return_conditional_losses_6020

inputs1
-stream_readvariableop_streaming_stream_states<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias5
1stream_1_readvariableop_streaming_stream_1_statesD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel5
1stream_2_readvariableop_streaming_stream_2_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpขstream_1/AssignVariableOpขstream_2/AssignVariableOp
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimฬ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(2#
!tf_op_layer_ExpandDims/ExpandDimsค
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02
stream/ReadVariableOp
stream/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack
stream/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฎ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisฬ
stream/concatConcatV2stream/strided_slice:output:0*tf_op_layer_ExpandDims/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*&
_output_shapes
:(2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOpๅ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel^stream/AssignVariableOp*&
_output_shapes
:*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpี
stream/conv2d/Conv2DConv2Dstream/concat:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2
stream/conv2d/Conv2Dฺ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias^stream/AssignVariableOp*
_output_shapes
:*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOpท
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2
stream/conv2d/Reluฌ
stream_1/ReadVariableOpReadVariableOp1stream_1_readvariableop_streaming_stream_1_states*&
_output_shapes
:&*
dtype02
stream_1/ReadVariableOp
stream_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_1/strided_slice/stack
stream_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_1/strided_slice/stack_1
stream_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_1/strided_slice/stack_2บ
stream_1/strided_sliceStridedSlicestream_1/ReadVariableOp:value:0%stream_1/strided_slice/stack:output:0'stream_1/strided_slice/stack_1:output:0'stream_1/strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:&*

begin_mask*
end_mask2
stream_1/strided_slicen
stream_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_1/concat/axisส
stream_1/concatConcatV2stream_1/strided_slice:output:0 stream/conv2d/Relu:activations:0stream_1/concat/axis:output:0*
N*
T0*&
_output_shapes
:&2
stream_1/concatิ
stream_1/AssignVariableOpAssignVariableOp1stream_1_readvariableop_streaming_stream_1_statesstream_1/concat:output:0^stream_1/ReadVariableOp*
_output_shapes
 *
dtype02
stream_1/AssignVariableOp๗
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel^stream_1/AssignVariableOp*&
_output_shapes
:*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpใ
stream_1/conv2d_1/Conv2DConv2Dstream_1/concat:output:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:$*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2D์
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias^stream_1/AssignVariableOp*
_output_shapes
:*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpว
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:$2
stream_1/conv2d_1/Reluw
reshape/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      $      2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slice}
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
valueB :
?????????2
reshape/Reshape/shape/1u
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :ภ2
reshape/Reshape/shape/2ศ
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shapeก
reshape/ReshapeReshape$stream_1/conv2d_1/Relu:activations:0reshape/Reshape/shape:output:0*
T0*#
_output_shapes
:ภ2
reshape/Reshape
gru_1/SqueezeSqueezereshape/Reshape:output:0*
T0*
_output_shapes
:	ภ*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
gru_1/cell/unstackร
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	2
gru_1/cell/BiasAddf
gru_1/cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_1/cell/Const
gru_1/cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split/split_dimภ
gru_1/cell/splitSplit#gru_1/cell/split/split_dim:output:0gru_1/cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
gru_1/cell/Const_1
gru_1/cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
gru_1/cell/split_1/split_dim๒
gru_1/cell/split_1SplitVgru_1/cell/BiasAdd_1:output:0gru_1/cell/Const_1:output:0%gru_1/cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_1i
gru_1/cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_1/cell/sub/x
gru_1/cell/subSubgru_1/cell/sub/x:output:0gru_1/cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_3?
gru_1/AssignVariableOpAssignVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_stategru_1/cell/add_3:z:0#^gru_1/cell/MatMul_1/ReadVariableOp ^gru_1/cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru_1/AssignVariableOp
gru_1/ExpandDims/dimConst^gru_1/AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
gru_1/ExpandDims/dim
gru_1/ExpandDims
ExpandDimsgru_1/cell/add_3:z:0gru_1/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru_1/ExpandDimsฉ
stream_2/ReadVariableOpReadVariableOp1stream_2_readvariableop_streaming_stream_2_states*#
_output_shapes
:*
dtype02
stream_2/ReadVariableOp
stream_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
stream_2/strided_slice/stack
stream_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2 
stream_2/strided_slice/stack_1
stream_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2 
stream_2/strided_slice/stack_2ต
stream_2/strided_sliceStridedSlicestream_2/ReadVariableOp:value:0%stream_2/strided_slice/stack:output:0'stream_2/strided_slice/stack_1:output:0'stream_2/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream_2/strided_slicen
stream_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream_2/concat/axisภ
stream_2/concatConcatV2stream_2/strided_slice:output:0gru_1/ExpandDims:output:0stream_2/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream_2/concatิ
stream_2/AssignVariableOpAssignVariableOp1stream_2_readvariableop_streaming_stream_2_statesstream_2/concat:output:0^stream_2/ReadVariableOp*
_output_shapes
 *
dtype02
stream_2/AssignVariableOp
stream_2/flatten/ConstConst^stream_2/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????   2
stream_2/flatten/Constค
stream_2/flatten/ReshapeReshapestream_2/concat:output:0stream_2/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_2/flatten/Reshape}
dropout/IdentityIdentity!stream_2/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identityฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMulซ
dense/BiasAdd/ReadVariableOpReadVariableOp1dense_biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAddท
dense_1/MatMul/ReadVariableOpReadVariableOp6dense_1_matmul_readvariableop_streaming_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMulณ
dense_1/BiasAdd/ReadVariableOpReadVariableOp5dense_1_biasadd_readvariableop_streaming_dense_1_bias*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/BiasAddh
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
dense_1/Reluถ
dense_2/MatMul/ReadVariableOpReadVariableOp6dense_2_matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMulฒ
dense_2/BiasAdd/ReadVariableOpReadVariableOp5dense_2_biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAddฮ
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp^stream_1/AssignVariableOp^stream_2/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*e
_input_shapesT
R:(:::::::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp26
stream_1/AssignVariableOpstream_1/AssignVariableOp26
stream_2/AssignVariableOpstream_2/AssignVariableOp:S O
+
_output_shapes
:?????????(
 
_user_specified_nameinputs
ใ&
โ
?__inference_gru_1_layer_call_and_return_conditional_losses_5474

inputs1
-cell_readvariableop_streaming_gru_1_cell_bias:
6cell_matmul_readvariableop_streaming_gru_1_cell_kernel6
2cell_matmul_1_readvariableop_streaming_input_stateH
Dcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel
identityขAssignVariableOpf
SqueezeSqueezeinputs*
T0*
_output_shapes
:	ภ*
squeeze_dims
2	
Squeeze
cell/ReadVariableOpReadVariableOp-cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
cell/ReadVariableOp{
cell/unstackUnpackcell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/unstackฑ
cell/MatMul/ReadVariableOpReadVariableOp6cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02
cell/MatMul/ReadVariableOp
cell/MatMulMatMulSqueeze:output:0"cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/MatMul
cell/BiasAddBiasAddcell/MatMul:product:0cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/BiasAddZ

cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

cell/Constw
cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/split/split_dimจ

cell/splitSplitcell/split/split_dim:output:0cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2

cell/splitฐ
cell/MatMul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/MatMul_1/ReadVariableOpว
cell/MatMul_1/ReadVariableOp_1ReadVariableOpDcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02 
cell/MatMul_1/ReadVariableOp_1?
cell/MatMul_1MatMul$cell/MatMul_1/ReadVariableOp:value:0&cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
cell/MatMul_1
cell/BiasAdd_1BiasAddcell/MatMul_1:product:0cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/BiasAdd_1q
cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/Const_1{
cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/split_1/split_dimิ
cell/split_1SplitVcell/BiasAdd_1:output:0cell/Const_1:output:0cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/split_1s
cell/addAddV2cell/split:output:0cell/split_1:output:0*
T0*
_output_shapes
:	2

cell/add_
cell/SigmoidSigmoidcell/add:z:0*
T0*
_output_shapes
:	2
cell/Sigmoidw

cell/add_1AddV2cell/split:output:1cell/split_1:output:1*
T0*
_output_shapes
:	2

cell/add_1e
cell/Sigmoid_1Sigmoidcell/add_1:z:0*
T0*
_output_shapes
:	2
cell/Sigmoid_1p
cell/mulMulcell/Sigmoid_1:y:0cell/split_1:output:2*
T0*
_output_shapes
:	2

cell/muln

cell/add_2AddV2cell/split:output:2cell/mul:z:0*
T0*
_output_shapes
:	2

cell/add_2X
	cell/TanhTanhcell/add_2:z:0*
T0*
_output_shapes
:	2
	cell/Tanhช
cell/mul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/mul_1/ReadVariableOp~

cell/mul_1Mulcell/Sigmoid:y:0!cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

cell/mul_1]

cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cell/sub/xl
cell/subSubcell/sub/x:output:0cell/Sigmoid:y:0*
T0*
_output_shapes
:	2

cell/subf

cell/mul_2Mulcell/sub:z:0cell/Tanh:y:0*
T0*
_output_shapes
:	2

cell/mul_2k

cell/add_3AddV2cell/mul_1:z:0cell/mul_2:z:0*
T0*
_output_shapes
:	2

cell/add_3ฺ
AssignVariableOpAssignVariableOp2cell_matmul_1_readvariableop_streaming_input_statecell/add_3:z:0^cell/MatMul_1/ReadVariableOp^cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpu
ExpandDims/dimConst^AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim}

ExpandDims
ExpandDimscell/add_3:z:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDimsv
IdentityIdentityExpandDims:output:0^AssignVariableOp*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ภ::::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:ภ
 
_user_specified_nameinputs
ใ&
โ
?__inference_gru_1_layer_call_and_return_conditional_losses_6472

inputs1
-cell_readvariableop_streaming_gru_1_cell_bias:
6cell_matmul_readvariableop_streaming_gru_1_cell_kernel6
2cell_matmul_1_readvariableop_streaming_input_stateH
Dcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel
identityขAssignVariableOpf
SqueezeSqueezeinputs*
T0*
_output_shapes
:	ภ*
squeeze_dims
2	
Squeeze
cell/ReadVariableOpReadVariableOp-cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	*
dtype02
cell/ReadVariableOp{
cell/unstackUnpackcell/ReadVariableOp:value:0*
T0*"
_output_shapes
::*	
num2
cell/unstackฑ
cell/MatMul/ReadVariableOpReadVariableOp6cell_matmul_readvariableop_streaming_gru_1_cell_kernel* 
_output_shapes
:
ภ*
dtype02
cell/MatMul/ReadVariableOp
cell/MatMulMatMulSqueeze:output:0"cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/MatMul
cell/BiasAddBiasAddcell/MatMul:product:0cell/unstack:output:0*
T0*
_output_shapes
:	2
cell/BiasAddZ

cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2

cell/Constw
cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/split/split_dimจ

cell/splitSplitcell/split/split_dim:output:0cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2

cell/splitฐ
cell/MatMul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/MatMul_1/ReadVariableOpว
cell/MatMul_1/ReadVariableOp_1ReadVariableOpDcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
*
dtype02 
cell/MatMul_1/ReadVariableOp_1?
cell/MatMul_1MatMul$cell/MatMul_1/ReadVariableOp:value:0&cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	2
cell/MatMul_1
cell/BiasAdd_1BiasAddcell/MatMul_1:product:0cell/unstack:output:1*
T0*
_output_shapes
:	2
cell/BiasAdd_1q
cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"      ????2
cell/Const_1{
cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cell/split_1/split_dimิ
cell/split_1SplitVcell/BiasAdd_1:output:0cell/Const_1:output:0cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/split_1s
cell/addAddV2cell/split:output:0cell/split_1:output:0*
T0*
_output_shapes
:	2

cell/add_
cell/SigmoidSigmoidcell/add:z:0*
T0*
_output_shapes
:	2
cell/Sigmoidw

cell/add_1AddV2cell/split:output:1cell/split_1:output:1*
T0*
_output_shapes
:	2

cell/add_1e
cell/Sigmoid_1Sigmoidcell/add_1:z:0*
T0*
_output_shapes
:	2
cell/Sigmoid_1p
cell/mulMulcell/Sigmoid_1:y:0cell/split_1:output:2*
T0*
_output_shapes
:	2

cell/muln

cell/add_2AddV2cell/split:output:2cell/mul:z:0*
T0*
_output_shapes
:	2

cell/add_2X
	cell/TanhTanhcell/add_2:z:0*
T0*
_output_shapes
:	2
	cell/Tanhช
cell/mul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/mul_1/ReadVariableOp~

cell/mul_1Mulcell/Sigmoid:y:0!cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

cell/mul_1]

cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2

cell/sub/xl
cell/subSubcell/sub/x:output:0cell/Sigmoid:y:0*
T0*
_output_shapes
:	2

cell/subf

cell/mul_2Mulcell/sub:z:0cell/Tanh:y:0*
T0*
_output_shapes
:	2

cell/mul_2k

cell/add_3AddV2cell/mul_1:z:0cell/mul_2:z:0*
T0*
_output_shapes
:	2

cell/add_3ฺ
AssignVariableOpAssignVariableOp2cell_matmul_1_readvariableop_streaming_input_statecell/add_3:z:0^cell/MatMul_1/ReadVariableOp^cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpu
ExpandDims/dimConst^AssignVariableOp*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim}

ExpandDims
ExpandDimscell/add_3:z:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDimsv
IdentityIdentityExpandDims:output:0^AssignVariableOp*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*2
_input_shapes!
:ภ::::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:ภ
 
_user_specified_nameinputs
็

@__inference_stream_layer_call_and_return_conditional_losses_5318

inputs*
&readvariableop_streaming_stream_states5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identityขAssignVariableOp
ReadVariableOpReadVariableOp&readvariableop_streaming_stream_states*&
_output_shapes
:(*
dtype02
ReadVariableOp
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
strided_slice/stack_2
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*&
_output_shapes
:(*

begin_mask*
end_mask2
strided_slice\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2strided_slice:output:0inputsconcat/axis:output:0*
N*
T0*&
_output_shapes
:(2
concatฅ
AssignVariableOpAssignVariableOp&readvariableop_streaming_stream_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpษ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel^AssignVariableOp*&
_output_shapes
:*
dtype02
conv2d/Conv2D/ReadVariableOpน
conv2d/Conv2DConv2Dconcat:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:&*
paddingVALID*
strides
2
conv2d/Conv2Dพ
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias^AssignVariableOp*
_output_shapes
:*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:&2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:&2
conv2d/Relu
IdentityIdentityconv2d/Relu:activations:0^AssignVariableOp*
T0*&
_output_shapes
:&2

Identity"
identityIdentity:output:0*1
_input_shapes 
:(:::2$
AssignVariableOpAssignVariableOp:N J
&
_output_shapes
:(
 
_user_specified_nameinputs
?
ว
A__inference_dense_2_layer_call_and_return_conditional_losses_5608

inputs2
.matmul_readvariableop_streaming_dense_2_kernel1
-biasadd_readvariableop_streaming_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_streaming_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp-biasadd_readvariableop_streaming_dense_2_bias*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpx
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2	
BiasAdd[
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
์
B
&__inference_dropout_layer_call_fn_6581

inputs
identityท
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_55402
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ถ
เ
+__inference_functional_1_layer_call_fn_6323
input_audio
streaming_stream_states
stream_conv2d_kernel
stream_conv2d_bias
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_2_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCallซ
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_stream_statesstream_conv2d_kernelstream_conv2d_biasstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_2_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*/
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_56862
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????(:::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????(
%
_user_specified_nameinput_audio
ม

'__inference_stream_2_layer_call_fn_6554

inputs
streaming_stream_2_states
identityขStatefulPartitionedCall์
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_2_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_2_layer_call_and_return_conditional_losses_55132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:
 
_user_specified_nameinputs

Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_6356

inputs
identityอ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_52912
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:(2

Identity"
identityIdentity:output:0*!
_input_shapes
:(:J F
"
_output_shapes
:(
 
_user_specified_nameinputs
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_6351

inputs
identityk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:(2

Identity"
identityIdentity:output:0*!
_input_shapes
:(:J F
"
_output_shapes
:(
 
_user_specified_nameinputs
ึ
ม
?__inference_dense_layer_call_and_return_conditional_losses_6591

inputs0
,matmul_readvariableop_streaming_dense_kernel/
+biasadd_readvariableop_streaming_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp+biasadd_readvariableop_streaming_dense_bias*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOpy
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2	
BiasAdd\
IdentityIdentityBiasAdd:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
ะ
น
'__inference_stream_1_layer_call_fn_6410

inputs
streaming_stream_1_states
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
identityขStatefulPartitionedCallซ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_1_statesstream_1_conv2d_1_kernelstream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_stream_1_layer_call_and_return_conditional_losses_53512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:$2

Identity"
identityIdentity:output:0*1
_input_shapes 
:&:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:&
 
_user_specified_nameinputs"ธL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ค
serving_default
>
input_audio/
serving_default_input_audio:0(2
dense_2'
StatefulPartitionedCall:0tensorflow/serving/predict:ซ๕
]
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer-7
	layer_with_weights-4
	layer-8

layer_with_weights-5

layer-9
layer_with_weights-6
layer-10
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+ฌ&call_and_return_all_conditional_losses
ญ_default_save_signature
ฎ__call__"Y
_tf_keras_networkํX{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}, "name": "input_audio", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_audio", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 40, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 38, 16], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 576]}}, "name": "reshape", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "mode": "STREAM_INTERNAL_STATE_INFERENCE", "inference_batch_size": 1, "units": 256, "return_sequences": 0, "unroll": true, "stateful": true}, "name": "gru_1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 256], "ring_buffer_size_in_time_dim": 1}, "name": "stream_2", "inbound_nodes": [[["gru_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_audio", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}, "name": "input_audio", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_audio", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 40, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 38, 16], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 576]}}, "name": "reshape", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "mode": "STREAM_INTERNAL_STATE_INFERENCE", "inference_batch_size": 1, "units": 256, "return_sequences": 0, "unroll": true, "stateful": true}, "name": "gru_1", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 256], "ring_buffer_size_in_time_dim": 1}, "name": "stream_2", "inbound_nodes": [[["gru_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_audio", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
๓"๐
_tf_keras_input_layerะ{"class_name": "InputLayer", "name": "input_audio", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}}

	variables
regularization_losses
trainable_variables
	keras_api
+ฏ&call_and_return_all_conditional_losses
ฐ__call__"๚
_tf_keras_layerเ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"T": {"type": "DT_FLOAT"}, "Tdim": {"type": "DT_INT32"}}}, "constants": {"1": -1}}}

cell
state_shape

states
	variables
regularization_losses
trainable_variables
	keras_api
+ฑ&call_and_return_all_conditional_losses
ฒ__call__"ี	
_tf_keras_layerป	{"class_name": "Stream", "name": "stream", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 40, 1], "ring_buffer_size_in_time_dim": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 40, 1]}}

cell
state_shape

states
	variables
 regularization_losses
!trainable_variables
"	keras_api
+ณ&call_and_return_all_conditional_losses
ด__call__"?	
_tf_keras_layerร	{"class_name": "Stream", "name": "stream_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 38, 16], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 38, 16]}}
๕
#	variables
$regularization_losses
%trainable_variables
&	keras_api
+ต&call_and_return_all_conditional_losses
ถ__call__"ไ
_tf_keras_layerส{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [-1, 576]}}}
ื
'input_state
(gru_cell
)	variables
*regularization_losses
+trainable_variables
,	keras_api
+ท&call_and_return_all_conditional_losses
ธ__call__"ง
_tf_keras_layer{"class_name": "GRU", "name": "gru_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "mode": "STREAM_INTERNAL_STATE_INFERENCE", "inference_batch_size": 1, "units": 256, "return_sequences": 0, "unroll": true, "stateful": true}}
๎
-cell
.state_shape

/states
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+น&call_and_return_all_conditional_losses
บ__call__"ถ
_tf_keras_layer{"class_name": "Stream", "name": "stream_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 256], "ring_buffer_size_in_time_dim": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 256]}}
ใ
4	variables
5regularization_losses
6trainable_variables
7	keras_api
+ป&call_and_return_all_conditional_losses
ผ__call__"า
_tf_keras_layerธ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ฉ

8kernel
9bias
:	variables
;regularization_losses
<trainable_variables
=	keras_api
+ฝ&call_and_return_all_conditional_losses
พ__call__"
_tf_keras_layer่{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
ซ

>kernel
?bias
@	variables
Aregularization_losses
Btrainable_variables
C	keras_api
+ฟ&call_and_return_all_conditional_losses
ภ__call__"
_tf_keras_layer๊{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
ฌ

Dkernel
Ebias
F	variables
Gregularization_losses
Htrainable_variables
I	keras_api
+ม&call_and_return_all_conditional_losses
ย__call__"
_tf_keras_layer๋{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}

J0
K1
2
L3
M4
5
N6
O7
P8
'9
/10
811
912
>13
?14
D15
E16"
trackable_list_wrapper
 "
trackable_list_wrapper
~
J0
K1
L2
M3
N4
O5
P6
87
98
>9
?10
D11
E12"
trackable_list_wrapper
ฮ
Qlayer_metrics

Rlayers
	variables
Smetrics
Tnon_trainable_variables
regularization_losses
Ulayer_regularization_losses
trainable_variables
ฎ__call__
ญ_default_save_signature
+ฌ&call_and_return_all_conditional_losses
'ฌ"call_and_return_conditional_losses"
_generic_user_object
-
รserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
Vlayer_metrics

Wlayers
	variables
Xmetrics
Ynon_trainable_variables
regularization_losses
Zlayer_regularization_losses
trainable_variables
ฐ__call__
+ฏ&call_and_return_all_conditional_losses
'ฏ"call_and_return_conditional_losses"
_generic_user_object
	

Jkernel
Kbias
[	variables
\regularization_losses
]trainable_variables
^	keras_api
+ฤ&call_and_return_all_conditional_losses
ล__call__"๘
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}}
 "
trackable_list_wrapper
/:-(2streaming/stream/states
5
J0
K1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
ฐ
_layer_metrics

`layers
	variables
ametrics
bnon_trainable_variables
regularization_losses
clayer_regularization_losses
trainable_variables
ฒ__call__
+ฑ&call_and_return_all_conditional_losses
'ฑ"call_and_return_conditional_losses"
_generic_user_object
ค	

Lkernel
Mbias
d	variables
eregularization_losses
ftrainable_variables
g	keras_api
+ฦ&call_and_return_all_conditional_losses
ว__call__"?
_tf_keras_layerใ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}}
 "
trackable_list_wrapper
1:/&2streaming/stream_1/states
5
L0
M1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
ฐ
hlayer_metrics

ilayers
	variables
jmetrics
knon_trainable_variables
 regularization_losses
llayer_regularization_losses
!trainable_variables
ด__call__
+ณ&call_and_return_all_conditional_losses
'ณ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ฐ
mlayer_metrics

nlayers
#	variables
ometrics
pnon_trainable_variables
$regularization_losses
qlayer_regularization_losses
%trainable_variables
ถ__call__
+ต&call_and_return_all_conditional_losses
'ต"call_and_return_conditional_losses"
_generic_user_object
&:$	2streaming/input_state


Nkernel
Orecurrent_kernel
Pbias
r	variables
sregularization_losses
ttrainable_variables
u	keras_api
+ศ&call_and_return_all_conditional_losses
ษ__call__"ฺ
_tf_keras_layerภ{"class_name": "GRUCell", "name": "cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cell", "trainable": true, "dtype": "float32", "units": 256, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
<
N0
O1
P2
'3"
trackable_list_wrapper
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
ฐ
vlayer_metrics

wlayers
)	variables
xmetrics
ynon_trainable_variables
*regularization_losses
zlayer_regularization_losses
+trainable_variables
ธ__call__
+ท&call_and_return_all_conditional_losses
'ท"call_and_return_conditional_losses"
_generic_user_object
ไ
{	variables
|regularization_losses
}trainable_variables
~	keras_api
+ส&call_and_return_all_conditional_losses
ห__call__"ำ
_tf_keras_layerน{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
.:,2streaming/stream_2/states
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ด
layer_metrics
layers
0	variables
metrics
non_trainable_variables
1regularization_losses
 layer_regularization_losses
2trainable_variables
บ__call__
+น&call_and_return_all_conditional_losses
'น"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
layer_metrics
layers
4	variables
metrics
non_trainable_variables
5regularization_losses
 layer_regularization_losses
6trainable_variables
ผ__call__
+ป&call_and_return_all_conditional_losses
'ป"call_and_return_conditional_losses"
_generic_user_object
*:(
2streaming/dense/kernel
#:!2streaming/dense/bias
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
ต
layer_metrics
layers
:	variables
metrics
non_trainable_variables
;regularization_losses
 layer_regularization_losses
<trainable_variables
พ__call__
+ฝ&call_and_return_all_conditional_losses
'ฝ"call_and_return_conditional_losses"
_generic_user_object
,:*
2streaming/dense_1/kernel
%:#2streaming/dense_1/bias
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
ต
layer_metrics
layers
@	variables
metrics
non_trainable_variables
Aregularization_losses
 layer_regularization_losses
Btrainable_variables
ภ__call__
+ฟ&call_and_return_all_conditional_losses
'ฟ"call_and_return_conditional_losses"
_generic_user_object
+:)	2streaming/dense_2/kernel
$:"2streaming/dense_2/bias
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
ต
layer_metrics
layers
F	variables
metrics
non_trainable_variables
Gregularization_losses
 layer_regularization_losses
Htrainable_variables
ย__call__
+ม&call_and_return_all_conditional_losses
'ม"call_and_return_conditional_losses"
_generic_user_object
.:,2stream/conv2d/kernel
 :2stream/conv2d/bias
2:02stream_1/conv2d_1/kernel
$:"2stream_1/conv2d_1/bias
/:-
ภ2streaming/gru_1/cell/kernel
9:7
2%streaming/gru_1/cell/recurrent_kernel
,:*	2streaming/gru_1/cell/bias
 "
trackable_dict_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
'2
/3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
ต
layer_metrics
layers
[	variables
metrics
non_trainable_variables
\regularization_losses
 layer_regularization_losses
]trainable_variables
ล__call__
+ฤ&call_and_return_all_conditional_losses
'ฤ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
ต
layer_metrics
layers
d	variables
metrics
?non_trainable_variables
eregularization_losses
 กlayer_regularization_losses
ftrainable_variables
ว__call__
+ฦ&call_and_return_all_conditional_losses
'ฦ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
 "
trackable_list_wrapper
5
N0
O1
P2"
trackable_list_wrapper
ต
ขlayer_metrics
ฃlayers
r	variables
คmetrics
ฅnon_trainable_variables
sregularization_losses
 ฆlayer_regularization_losses
ttrainable_variables
ษ__call__
+ศ&call_and_return_all_conditional_losses
'ศ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
(0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
'0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ต
งlayer_metrics
จlayers
{	variables
ฉmetrics
ชnon_trainable_variables
|regularization_losses
 ซlayer_regularization_losses
}trainable_variables
ห__call__
+ส&call_and_return_all_conditional_losses
'ส"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
-0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
/0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ๆ2ใ
F__inference_functional_1_layer_call_and_return_conditional_losses_5905
F__inference_functional_1_layer_call_and_return_conditional_losses_6186
F__inference_functional_1_layer_call_and_return_conditional_losses_6301
F__inference_functional_1_layer_call_and_return_conditional_losses_6020ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ๅ2โ
__inference__wrapped_model_5281พ
ฒ
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *.ข+
)&
input_audio?????????(
๚2๗
+__inference_functional_1_layer_call_fn_6064
+__inference_functional_1_layer_call_fn_6345
+__inference_functional_1_layer_call_fn_6323
+__inference_functional_1_layer_call_fn_6042ภ
ทฒณ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
๚2๗
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_6351ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
฿2?
5__inference_tf_op_layer_ExpandDims_layer_call_fn_6356ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๊2็
@__inference_stream_layer_call_and_return_conditional_losses_6375ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฯ2ฬ
%__inference_stream_layer_call_fn_6383ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_stream_1_layer_call_and_return_conditional_losses_6402ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_stream_1_layer_call_fn_6410ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_reshape_layer_call_and_return_conditional_losses_6423ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ะ2อ
&__inference_reshape_layer_call_fn_6428ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ป2ธ
?__inference_gru_1_layer_call_and_return_conditional_losses_6472
?__inference_gru_1_layer_call_and_return_conditional_losses_6516ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
2
$__inference_gru_1_layer_call_fn_6525
$__inference_gru_1_layer_call_fn_6534ณ
ชฒฆ
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
์2้
B__inference_stream_2_layer_call_and_return_conditional_losses_6548ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ั2ฮ
'__inference_stream_2_layer_call_fn_6554ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ภ2ฝ
A__inference_dropout_layer_call_and_return_conditional_losses_6571
A__inference_dropout_layer_call_and_return_conditional_losses_6566ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
2
&__inference_dropout_layer_call_fn_6581
&__inference_dropout_layer_call_fn_6576ด
ซฒง
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
้2ๆ
?__inference_dense_layer_call_and_return_conditional_losses_6591ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฮ2ห
$__inference_dense_layer_call_fn_6598ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_dense_1_layer_call_and_return_conditional_losses_6609ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ะ2อ
&__inference_dense_1_layer_call_fn_6616ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
๋2่
A__inference_dense_2_layer_call_and_return_conditional_losses_6626ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ะ2อ
&__inference_dense_2_layer_call_fn_6633ข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
5B3
"__inference_signature_wrapper_5783input_audio
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
ฤ2มพ
ตฒฑ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
ฤ2มพ
ตฒฑ
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsช 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
จ2ฅข
ฒ
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsช *
 
__inference__wrapped_model_5281wJKLMPN'O/89>?DE8ข5
.ข+
)&
input_audio?????????(
ช "(ช%
#
dense_2
dense_2
A__inference_dense_1_layer_call_and_return_conditional_losses_6609L>?'ข$
ข

inputs	
ช "ข

0	
 i
&__inference_dense_1_layer_call_fn_6616?>?'ข$
ข

inputs	
ช "	
A__inference_dense_2_layer_call_and_return_conditional_losses_6626KDE'ข$
ข

inputs	
ช "ข

0
 h
&__inference_dense_2_layer_call_fn_6633>DE'ข$
ข

inputs	
ช "
?__inference_dense_layer_call_and_return_conditional_losses_6591L89'ข$
ข

inputs	
ช "ข

0	
 g
$__inference_dense_layer_call_fn_6598?89'ข$
ข

inputs	
ช "	
A__inference_dropout_layer_call_and_return_conditional_losses_6566L+ข(
!ข

inputs	
p
ช "ข

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_6571L+ข(
!ข

inputs	
p 
ช "ข

0	
 i
&__inference_dropout_layer_call_fn_6576?+ข(
!ข

inputs	
p
ช "	i
&__inference_dropout_layer_call_fn_6581?+ข(
!ข

inputs	
p 
ช "	ธ
F__inference_functional_1_layer_call_and_return_conditional_losses_5905nJKLMPN'O/89>?DE;ข8
1ข.
$!
inputs?????????(
p

 
ช "ข

0
 ธ
F__inference_functional_1_layer_call_and_return_conditional_losses_6020nJKLMPN'O/89>?DE;ข8
1ข.
$!
inputs?????????(
p 

 
ช "ข

0
 ฝ
F__inference_functional_1_layer_call_and_return_conditional_losses_6186sJKLMPN'O/89>?DE@ข=
6ข3
)&
input_audio?????????(
p

 
ช "ข

0
 ฝ
F__inference_functional_1_layer_call_and_return_conditional_losses_6301sJKLMPN'O/89>?DE@ข=
6ข3
)&
input_audio?????????(
p 

 
ช "ข

0
 
+__inference_functional_1_layer_call_fn_6042aJKLMPN'O/89>?DE;ข8
1ข.
$!
inputs?????????(
p

 
ช "
+__inference_functional_1_layer_call_fn_6064aJKLMPN'O/89>?DE;ข8
1ข.
$!
inputs?????????(
p 

 
ช "
+__inference_functional_1_layer_call_fn_6323fJKLMPN'O/89>?DE@ข=
6ข3
)&
input_audio?????????(
p

 
ช "
+__inference_functional_1_layer_call_fn_6345fJKLMPN'O/89>?DE@ข=
6ข3
)&
input_audio?????????(
p 

 
ช "
?__inference_gru_1_layer_call_and_return_conditional_losses_6472ZPN'O/ข,
%ข"

inputsภ
p
ช "!ข

0
 
?__inference_gru_1_layer_call_and_return_conditional_losses_6516ZPN'O/ข,
%ข"

inputsภ
p 
ช "!ข

0
 u
$__inference_gru_1_layer_call_fn_6525MPN'O/ข,
%ข"

inputsภ
p
ช "u
$__inference_gru_1_layer_call_fn_6534MPN'O/ข,
%ข"

inputsภ
p 
ช "
A__inference_reshape_layer_call_and_return_conditional_losses_6423S.ข+
$ข!

inputs$
ช "!ข

0ภ
 p
&__inference_reshape_layer_call_fn_6428F.ข+
$ข!

inputs$
ช "ภฃ
"__inference_signature_wrapper_5783}JKLMPN'O/89>?DE>ข;
ข 
4ช1
/
input_audio 
input_audio("(ช%
#
dense_2
dense_2ก
B__inference_stream_1_layer_call_and_return_conditional_losses_6402[LM.ข+
$ข!

inputs&
ช "$ข!

0$
 y
'__inference_stream_1_layer_call_fn_6410NLM.ข+
$ข!

inputs&
ช "$
B__inference_stream_2_layer_call_and_return_conditional_losses_6548O/+ข(
!ข

inputs
ช "ข

0	
 m
'__inference_stream_2_layer_call_fn_6554B/+ข(
!ข

inputs
ช "	
@__inference_stream_layer_call_and_return_conditional_losses_6375[JK.ข+
$ข!

inputs(
ช "$ข!

0&
 w
%__inference_stream_layer_call_fn_6383NJK.ข+
$ข!

inputs(
ช "&ฆ
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_6351R*ข'
 ข

inputs(
ช "$ข!

0(
 ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_6356E*ข'
 ข

inputs(
ช "(