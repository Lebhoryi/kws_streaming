งอ	
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878๋

streaming/input_stateVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_namestreaming/input_state

)streaming/input_state/Read/ReadVariableOpReadVariableOpstreaming/input_state*
_output_shapes
:	*
dtype0

streaming/stream/statesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namestreaming/stream/states

+streaming/stream/states/Read/ReadVariableOpReadVariableOpstreaming/stream/states*#
_output_shapes
:*
dtype0

streaming/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_namestreaming/dense/kernel

*streaming/dense/kernel/Read/ReadVariableOpReadVariableOpstreaming/dense/kernel* 
_output_shapes
:
*
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

streaming/gru_1/cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ฐ	*,
shared_namestreaming/gru_1/cell/kernel

/streaming/gru_1/cell/kernel/Read/ReadVariableOpReadVariableOpstreaming/gru_1/cell/kernel*
_output_shapes
:	ฐ	*
dtype0
จ
%streaming/gru_1/cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ฐ	*6
shared_name'%streaming/gru_1/cell/recurrent_kernel
ก
9streaming/gru_1/cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp%streaming/gru_1/cell/recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype0

streaming/gru_1/cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ฐ	**
shared_namestreaming/gru_1/cell/bias

-streaming/gru_1/cell/bias/Read/ReadVariableOpReadVariableOpstreaming/gru_1/cell/bias*
_output_shapes
:	ฐ	*
dtype0

NoOpNoOp
ฅ#
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*เ"
valueึ"Bำ" Bฬ"
ฟ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
 
q
input_state
gru_cell
trainable_variables
regularization_losses
	variables
	keras_api
y
cell
state_shape

states
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
h

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
?
00
11
22
3
4
$5
%6
*7
+8
 
N
00
11
22
3
4
5
6
$7
%8
*9
+10
ญ
3layer_metrics

4layers
trainable_variables
5layer_regularization_losses
6metrics
7non_trainable_variables
	regularization_losses

	variables
 
fd
VARIABLE_VALUEstreaming/input_state;layer_with_weights-0/input_state/.ATTRIBUTES/VARIABLE_VALUE
~

0kernel
1recurrent_kernel
2bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api

00
11
22
 

00
11
22
3
ญ
<layer_metrics

=layers
trainable_variables
>layer_regularization_losses
?metrics
@non_trainable_variables
regularization_losses
	variables
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
 
ca
VARIABLE_VALUEstreaming/stream/states6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
ญ
Elayer_metrics

Flayers
trainable_variables
Glayer_regularization_losses
Hmetrics
Inon_trainable_variables
regularization_losses
	variables
 
 
 
ญ
Jlayer_metrics

Klayers
trainable_variables
Llayer_regularization_losses
Mmetrics
Nnon_trainable_variables
regularization_losses
	variables
b`
VARIABLE_VALUEstreaming/dense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEstreaming/dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
ญ
Olayer_metrics

Players
 trainable_variables
Qlayer_regularization_losses
Rmetrics
Snon_trainable_variables
!regularization_losses
"	variables
db
VARIABLE_VALUEstreaming/dense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstreaming/dense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
ญ
Tlayer_metrics

Ulayers
&trainable_variables
Vlayer_regularization_losses
Wmetrics
Xnon_trainable_variables
'regularization_losses
(	variables
db
VARIABLE_VALUEstreaming/dense_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstreaming/dense_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

*0
+1
 

*0
+1
ญ
Ylayer_metrics

Zlayers
,trainable_variables
[layer_regularization_losses
\metrics
]non_trainable_variables
-regularization_losses
.	variables
a_
VARIABLE_VALUEstreaming/gru_1/cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%streaming/gru_1/cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEstreaming/gru_1/cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6
 
 

0
1

00
11
22
 

00
11
22
ญ
^layer_metrics

_layers
8trainable_variables
`layer_regularization_losses
ametrics
bnon_trainable_variables
9regularization_losses
:	variables
 

0
 
 

0
 
 
 
ญ
clayer_metrics

dlayers
Atrainable_variables
elayer_regularization_losses
fmetrics
gnon_trainable_variables
Bregularization_losses
C	variables
 

0
 
 

0
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
:*
dtype0*
shape:
ๅ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_audiostreaming/gru_1/cell/biasstreaming/gru_1/cell/kernelstreaming/input_state%streaming/gru_1/cell/recurrent_kernelstreaming/stream/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstreaming/dense_2/kernelstreaming/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference_signature_wrapper_6500
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ฆ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)streaming/input_state/Read/ReadVariableOp+streaming/stream/states/Read/ReadVariableOp*streaming/dense/kernel/Read/ReadVariableOp(streaming/dense/bias/Read/ReadVariableOp,streaming/dense_1/kernel/Read/ReadVariableOp*streaming/dense_1/bias/Read/ReadVariableOp,streaming/dense_2/kernel/Read/ReadVariableOp*streaming/dense_2/bias/Read/ReadVariableOp/streaming/gru_1/cell/kernel/Read/ReadVariableOp9streaming/gru_1/cell/recurrent_kernel/Read/ReadVariableOp-streaming/gru_1/cell/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *&
f!R
__inference__traced_save_7135
ล
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestreaming/input_statestreaming/stream/statesstreaming/dense/kernelstreaming/dense/biasstreaming/dense_1/kernelstreaming/dense_1/biasstreaming/dense_2/kernelstreaming/dense_2/biasstreaming/gru_1/cell/kernel%streaming/gru_1/cell/recurrent_kernelstreaming/gru_1/cell/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *)
f$R"
 __inference__traced_restore_7178
?
ว
A__inference_dense_2_layer_call_and_return_conditional_losses_6373

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
?
ว
A__inference_dense_2_layer_call_and_return_conditional_losses_7072

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
ค
_
A__inference_dropout_layer_call_and_return_conditional_losses_6305

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
X
ฑ
F__inference_functional_1_layer_call_and_return_conditional_losses_6768

inputs7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel1
-stream_readvariableop_streaming_stream_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpq
gru_1/SqueezeSqueezeinputs*
T0*
_output_shapes

:*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
gru_1/cell/unstackย
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
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
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
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
:2
gru_1/ExpandDimsก
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*#
_output_shapes
:*
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
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฉ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisธ
stream/concatConcatV2stream/strided_slice:output:0gru_1/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOp
stream/flatten/ConstConst^stream/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2
stream/flatten/Const
stream/flatten/ReshapeReshapestream/concat:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ไ8?2
dropout/dropout/Const
dropout/dropout/MulMulstream/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/dropout/Shapeฤ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
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
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1ฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
฿&
โ
?__inference_gru_1_layer_call_and_return_conditional_losses_6918

inputs1
-cell_readvariableop_streaming_gru_1_cell_bias:
6cell_matmul_readvariableop_streaming_gru_1_cell_kernel6
2cell_matmul_1_readvariableop_streaming_input_stateH
Dcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel
identityขAssignVariableOpe
SqueezeSqueezeinputs*
T0*
_output_shapes

:*
squeeze_dims
2	
Squeeze
cell/ReadVariableOpReadVariableOp-cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
cell/ReadVariableOp{
cell/unstackUnpackcell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
cell/unstackฐ
cell/MatMul/ReadVariableOpReadVariableOp6cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02
cell/MatMul/ReadVariableOp
cell/MatMulMatMulSqueeze:output:0"cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
cell/MatMul
cell/BiasAddBiasAddcell/MatMul:product:0cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2

cell/splitฐ
cell/MatMul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/MatMul_1/ReadVariableOpว
cell/MatMul_1/ReadVariableOp_1ReadVariableOpDcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02 
cell/MatMul_1/ReadVariableOp_1?
cell/MatMul_1MatMul$cell/MatMul_1/ReadVariableOp:value:0&cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
cell/MatMul_1
cell/BiasAdd_1BiasAddcell/MatMul_1:product:0cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
cell/BiasAdd_1q
cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
cell/split_1s
cell/addAddV2cell/split:output:0cell/split_1:output:0*
T0*
_output_shapes
:	2

cell/add_
cell/SigmoidSigmoidcell/add:z:0*
T0*
_output_shapes
:	2
cell/Sigmoidw

cell/add_1AddV2cell/split:output:1cell/split_1:output:1*
T0*
_output_shapes
:	2

cell/add_1e
cell/Sigmoid_1Sigmoidcell/add_1:z:0*
T0*
_output_shapes
:	2
cell/Sigmoid_1p
cell/mulMulcell/Sigmoid_1:y:0cell/split_1:output:2*
T0*
_output_shapes
:	2

cell/muln

cell/add_2AddV2cell/split:output:2cell/mul:z:0*
T0*
_output_shapes
:	2

cell/add_2X
	cell/TanhTanhcell/add_2:z:0*
T0*
_output_shapes
:	2
	cell/Tanhช
cell/mul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/mul_1/ReadVariableOp~

cell/mul_1Mulcell/Sigmoid:y:0!cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

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
:	2

cell/subf

cell/mul_2Mulcell/sub:z:0cell/Tanh:y:0*
T0*
_output_shapes
:	2

cell/mul_2k

cell/add_3AddV2cell/mul_1:z:0cell/mul_2:z:0*
T0*
_output_shapes
:	2

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
:2

ExpandDimsv
IdentityIdentityExpandDims:output:0^AssignVariableOp*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:::::2$
AssignVariableOpAssignVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
3
ส
 __inference__traced_restore_7178
file_prefix*
&assignvariableop_streaming_input_state.
*assignvariableop_1_streaming_stream_states-
)assignvariableop_2_streaming_dense_kernel+
'assignvariableop_3_streaming_dense_bias/
+assignvariableop_4_streaming_dense_1_kernel-
)assignvariableop_5_streaming_dense_1_bias/
+assignvariableop_6_streaming_dense_2_kernel-
)assignvariableop_7_streaming_dense_2_bias2
.assignvariableop_8_streaming_gru_1_cell_kernel<
8assignvariableop_9_streaming_gru_1_cell_recurrent_kernel1
-assignvariableop_10_streaming_gru_1_cell_bias
identity_12ขAssignVariableOpขAssignVariableOp_1ขAssignVariableOp_10ขAssignVariableOp_2ขAssignVariableOp_3ขAssignVariableOp_4ขAssignVariableOp_5ขAssignVariableOp_6ขAssignVariableOp_7ขAssignVariableOp_8ขAssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value?B๛B;layer_with_weights-0/input_state/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesฆ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices็
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*D
_output_shapes2
0::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityฅ
AssignVariableOpAssignVariableOp&assignvariableop_streaming_input_stateIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1ฏ
AssignVariableOp_1AssignVariableOp*assignvariableop_1_streaming_stream_statesIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2ฎ
AssignVariableOp_2AssignVariableOp)assignvariableop_2_streaming_dense_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3ฌ
AssignVariableOp_3AssignVariableOp'assignvariableop_3_streaming_dense_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ฐ
AssignVariableOp_4AssignVariableOp+assignvariableop_4_streaming_dense_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5ฎ
AssignVariableOp_5AssignVariableOp)assignvariableop_5_streaming_dense_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6ฐ
AssignVariableOp_6AssignVariableOp+assignvariableop_6_streaming_dense_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ฎ
AssignVariableOp_7AssignVariableOp)assignvariableop_7_streaming_dense_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ณ
AssignVariableOp_8AssignVariableOp.assignvariableop_8_streaming_gru_1_cell_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9ฝ
AssignVariableOp_9AssignVariableOp8assignvariableop_9_streaming_gru_1_cell_recurrent_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ต
AssignVariableOp_10AssignVariableOp-assignvariableop_10_streaming_gru_1_cell_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_109
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpะ
Identity_11Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_11ร
Identity_12IdentityIdentity_11:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_12"#
identity_12Identity_12:output:0*A
_input_shapes0
.: :::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102(
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
ุ

`
A__inference_dropout_layer_call_and_return_conditional_losses_7012

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
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/Shapeฌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
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
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
?
_
&__inference_dropout_layer_call_fn_7022

inputs
identityขStatefulPartitionedCallิ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_63002
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
?d
ธ
__inference__wrapped_model_6147
input_audioD
@functional_1_gru_1_cell_readvariableop_streaming_gru_1_cell_biasM
Ifunctional_1_gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernelI
Efunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state[
Wfunctional_1_gru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel>
:functional_1_stream_readvariableop_streaming_stream_statesC
?functional_1_dense_matmul_readvariableop_streaming_dense_kernelB
>functional_1_dense_biasadd_readvariableop_streaming_dense_biasG
Cfunctional_1_dense_1_matmul_readvariableop_streaming_dense_1_kernelF
Bfunctional_1_dense_1_biasadd_readvariableop_streaming_dense_1_biasG
Cfunctional_1_dense_2_matmul_readvariableop_streaming_dense_2_kernelF
Bfunctional_1_dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityข#functional_1/gru_1/AssignVariableOpข$functional_1/stream/AssignVariableOp
functional_1/gru_1/SqueezeSqueezeinput_audio*
T0*
_output_shapes

:*
squeeze_dims
2
functional_1/gru_1/Squeezeา
&functional_1/gru_1/cell/ReadVariableOpReadVariableOp@functional_1_gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02(
&functional_1/gru_1/cell/ReadVariableOpด
functional_1/gru_1/cell/unstackUnpack.functional_1/gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2!
functional_1/gru_1/cell/unstack้
-functional_1/gru_1/cell/MatMul/ReadVariableOpReadVariableOpIfunctional_1_gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02/
-functional_1/gru_1/cell/MatMul/ReadVariableOpะ
functional_1/gru_1/cell/MatMulMatMul#functional_1/gru_1/Squeeze:output:05functional_1/gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2 
functional_1/gru_1/cell/MatMulห
functional_1/gru_1/cell/BiasAddBiasAdd(functional_1/gru_1/cell/MatMul:product:0(functional_1/gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2!
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
!:	:	:	*
	num_split2
functional_1/gru_1/cell/split้
/functional_1/gru_1/cell/MatMul_1/ReadVariableOpReadVariableOpEfunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype021
/functional_1/gru_1/cell/MatMul_1/ReadVariableOp
1functional_1/gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpWfunctional_1_gru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype023
1functional_1/gru_1/cell/MatMul_1/ReadVariableOp_1์
 functional_1/gru_1/cell/MatMul_1MatMul7functional_1/gru_1/cell/MatMul_1/ReadVariableOp:value:09functional_1/gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2"
 functional_1/gru_1/cell/MatMul_1ั
!functional_1/gru_1/cell/BiasAdd_1BiasAdd*functional_1/gru_1/cell/MatMul_1:product:0(functional_1/gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2#
!functional_1/gru_1/cell/BiasAdd_1
functional_1/gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2!
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
!:	:	:	*
	num_split2!
functional_1/gru_1/cell/split_1ฟ
functional_1/gru_1/cell/addAddV2&functional_1/gru_1/cell/split:output:0(functional_1/gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add
functional_1/gru_1/cell/SigmoidSigmoidfunctional_1/gru_1/cell/add:z:0*
T0*
_output_shapes
:	2!
functional_1/gru_1/cell/Sigmoidร
functional_1/gru_1/cell/add_1AddV2&functional_1/gru_1/cell/split:output:1(functional_1/gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add_1
!functional_1/gru_1/cell/Sigmoid_1Sigmoid!functional_1/gru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2#
!functional_1/gru_1/cell/Sigmoid_1ผ
functional_1/gru_1/cell/mulMul%functional_1/gru_1/cell/Sigmoid_1:y:0(functional_1/gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/mulบ
functional_1/gru_1/cell/add_2AddV2&functional_1/gru_1/cell/split:output:2functional_1/gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/add_2
functional_1/gru_1/cell/TanhTanh!functional_1/gru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/Tanhใ
,functional_1/gru_1/cell/mul_1/ReadVariableOpReadVariableOpEfunctional_1_gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02.
,functional_1/gru_1/cell/mul_1/ReadVariableOpส
functional_1/gru_1/cell/mul_1Mul#functional_1/gru_1/cell/Sigmoid:y:04functional_1/gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
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
:	2
functional_1/gru_1/cell/subฒ
functional_1/gru_1/cell/mul_2Mulfunctional_1/gru_1/cell/sub:z:0 functional_1/gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
functional_1/gru_1/cell/mul_2ท
functional_1/gru_1/cell/add_3AddV2!functional_1/gru_1/cell/mul_1:z:0!functional_1/gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
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
:2
functional_1/gru_1/ExpandDimsศ
"functional_1/stream/ReadVariableOpReadVariableOp:functional_1_stream_readvariableop_streaming_stream_states*#
_output_shapes
:*
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
valueB"           2+
)functional_1/stream/strided_slice/stack_1ซ
)functional_1/stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2+
)functional_1/stream/strided_slice/stack_2๗
!functional_1/stream/strided_sliceStridedSlice*functional_1/stream/ReadVariableOp:value:00functional_1/stream/strided_slice/stack:output:02functional_1/stream/strided_slice/stack_1:output:02functional_1/stream/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2#
!functional_1/stream/strided_slice
functional_1/stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/stream/concat/axis๙
functional_1/stream/concatConcatV2*functional_1/stream/strided_slice:output:0&functional_1/gru_1/ExpandDims:output:0(functional_1/stream/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
functional_1/stream/concat
$functional_1/stream/AssignVariableOpAssignVariableOp:functional_1_stream_readvariableop_streaming_stream_states#functional_1/stream/concat:output:0#^functional_1/stream/ReadVariableOp*
_output_shapes
 *
dtype02&
$functional_1/stream/AssignVariableOpพ
!functional_1/stream/flatten/ConstConst%^functional_1/stream/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2#
!functional_1/stream/flatten/Constะ
#functional_1/stream/flatten/ReshapeReshape#functional_1/stream/concat:output:0*functional_1/stream/flatten/Const:output:0*
T0*
_output_shapes
:	2%
#functional_1/stream/flatten/Reshapeข
functional_1/dropout/IdentityIdentity,functional_1/stream/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/Identityึ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp?functional_1_dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
functional_1/dense_2/BiasAddฝ
IdentityIdentity%functional_1/dense_2/BiasAdd:output:0$^functional_1/gru_1/AssignVariableOp%^functional_1/stream/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::2J
#functional_1/gru_1/AssignVariableOp#functional_1/gru_1/AssignVariableOp2L
$functional_1/stream/AssignVariableOp$functional_1/stream/AssignVariableOp:X T
+
_output_shapes
:?????????
%
_user_specified_nameinput_audio

ใ
$__inference_gru_1_layer_call_fn_6980

inputs
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
identityขStatefulPartitionedCallี
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_62392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs
รO
ถ
F__inference_functional_1_layer_call_and_return_conditional_losses_6655
input_audio7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel1
-stream_readvariableop_streaming_stream_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpv
gru_1/SqueezeSqueezeinput_audio*
T0*
_output_shapes

:*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
gru_1/cell/unstackย
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
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
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
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
:2
gru_1/ExpandDimsก
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*#
_output_shapes
:*
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
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฉ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisธ
stream/concatConcatV2stream/strided_slice:output:0gru_1/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOp
stream/flatten/ConstConst^stream/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2
stream/flatten/Const
stream/flatten/ReshapeReshapestream/concat:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshape{
dropout/IdentityIdentitystream/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identityฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp:X T
+
_output_shapes
:?????????
%
_user_specified_nameinput_audio
ฮ
ฑ
+__inference_functional_1_layer_call_fn_6858

inputs
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_64312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ฮ
ฑ
+__inference_functional_1_layer_call_fn_6874

inputs
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_64682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
ถ
+__inference_functional_1_layer_call_fn_6671
input_audio
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_64312
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameinput_audio
ชX
ถ
F__inference_functional_1_layer_call_and_return_conditional_losses_6581
input_audio7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel1
-stream_readvariableop_streaming_stream_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpv
gru_1/SqueezeSqueezeinput_audio*
T0*
_output_shapes

:*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
gru_1/cell/unstackย
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
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
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
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
:2
gru_1/ExpandDimsก
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*#
_output_shapes
:*
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
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฉ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisธ
stream/concatConcatV2stream/strided_slice:output:0gru_1/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOp
stream/flatten/ConstConst^stream/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2
stream/flatten/Const
stream/flatten/ReshapeReshapestream/concat:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ไ8?2
dropout/dropout/Const
dropout/dropout/MulMulstream/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/dropout/Shapeฤ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
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
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1ฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp:X T
+
_output_shapes
:?????????
%
_user_specified_nameinput_audio
ค
_
A__inference_dropout_layer_call_and_return_conditional_losses_7017

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs

ใ
$__inference_gru_1_layer_call_fn_6971

inputs
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
identityขStatefulPartitionedCallี
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_62392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:
 
_user_specified_nameinputs


@__inference_stream_layer_call_and_return_conditional_losses_6278

inputs*
&readvariableop_streaming_stream_states
identityขAssignVariableOp
ReadVariableOpReadVariableOp&readvariableop_streaming_stream_states*#
_output_shapes
:*
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
: *

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
:2
concatฅ
AssignVariableOpAssignVariableOp&readvariableop_streaming_stream_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshapew
IdentityIdentityflatten/Reshape:output:0^AssignVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:
 
_user_specified_nameinputs
ด
ว
A__inference_dense_1_layer_call_and_return_conditional_losses_6351

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
ึ
ม
?__inference_dense_layer_call_and_return_conditional_losses_6328

inputs0
,matmul_readvariableop_streaming_dense_kernel/
+biasadd_readvariableop_streaming_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
พ
{
%__inference_stream_layer_call_fn_7000

inputs
streaming_stream_states
identityขStatefulPartitionedCallํ
StatefulPartitionedCallStatefulPartitionedCallinputsstreaming_stream_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_62782
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::22
StatefulPartitionedCallStatefulPartitionedCall:K G
#
_output_shapes
:
 
_user_specified_nameinputs
ดO
ฑ
F__inference_functional_1_layer_call_and_return_conditional_losses_6842

inputs7
3gru_1_cell_readvariableop_streaming_gru_1_cell_bias@
<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel<
8gru_1_cell_matmul_1_readvariableop_streaming_input_stateN
Jgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel1
-stream_readvariableop_streaming_stream_states6
2dense_matmul_readvariableop_streaming_dense_kernel5
1dense_biasadd_readvariableop_streaming_dense_bias:
6dense_1_matmul_readvariableop_streaming_dense_1_kernel9
5dense_1_biasadd_readvariableop_streaming_dense_1_bias:
6dense_2_matmul_readvariableop_streaming_dense_2_kernel9
5dense_2_biasadd_readvariableop_streaming_dense_2_bias
identityขgru_1/AssignVariableOpขstream/AssignVariableOpq
gru_1/SqueezeSqueezeinputs*
T0*
_output_shapes

:*
squeeze_dims
2
gru_1/Squeezeซ
gru_1/cell/ReadVariableOpReadVariableOp3gru_1_cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
gru_1/cell/ReadVariableOp
gru_1/cell/unstackUnpack!gru_1/cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
gru_1/cell/unstackย
 gru_1/cell/MatMul/ReadVariableOpReadVariableOp<gru_1_cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02"
 gru_1/cell/MatMul/ReadVariableOp
gru_1/cell/MatMulMatMulgru_1/Squeeze:output:0(gru_1/cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul
gru_1/cell/BiasAddBiasAddgru_1/cell/MatMul:product:0gru_1/cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2
gru_1/cell/splitย
"gru_1/cell/MatMul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02$
"gru_1/cell/MatMul_1/ReadVariableOpู
$gru_1/cell/MatMul_1/ReadVariableOp_1ReadVariableOpJgru_1_cell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02&
$gru_1/cell/MatMul_1/ReadVariableOp_1ธ
gru_1/cell/MatMul_1MatMul*gru_1/cell/MatMul_1/ReadVariableOp:value:0,gru_1/cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/MatMul_1
gru_1/cell/BiasAdd_1BiasAddgru_1/cell/MatMul_1:product:0gru_1/cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
gru_1/cell/BiasAdd_1}
gru_1/cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
gru_1/cell/split_1
gru_1/cell/addAddV2gru_1/cell/split:output:0gru_1/cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_1/cell/addq
gru_1/cell/SigmoidSigmoidgru_1/cell/add:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid
gru_1/cell/add_1AddV2gru_1/cell/split:output:1gru_1/cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_1/cell/add_1w
gru_1/cell/Sigmoid_1Sigmoidgru_1/cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Sigmoid_1
gru_1/cell/mulMulgru_1/cell/Sigmoid_1:y:0gru_1/cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_1/cell/mul
gru_1/cell/add_2AddV2gru_1/cell/split:output:2gru_1/cell/mul:z:0*
T0*
_output_shapes
:	2
gru_1/cell/add_2j
gru_1/cell/TanhTanhgru_1/cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_1/cell/Tanhผ
gru_1/cell/mul_1/ReadVariableOpReadVariableOp8gru_1_cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02!
gru_1/cell/mul_1/ReadVariableOp
gru_1/cell/mul_1Mulgru_1/cell/Sigmoid:y:0'gru_1/cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
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
:	2
gru_1/cell/sub~
gru_1/cell/mul_2Mulgru_1/cell/sub:z:0gru_1/cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_1/cell/mul_2
gru_1/cell/add_3AddV2gru_1/cell/mul_1:z:0gru_1/cell/mul_2:z:0*
T0*
_output_shapes
:	2
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
:2
gru_1/ExpandDimsก
stream/ReadVariableOpReadVariableOp-stream_readvariableop_streaming_stream_states*#
_output_shapes
:*
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
valueB"           2
stream/strided_slice/stack_1
stream/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         2
stream/strided_slice/stack_2ฉ
stream/strided_sliceStridedSlicestream/ReadVariableOp:value:0#stream/strided_slice/stack:output:0%stream/strided_slice/stack_1:output:0%stream/strided_slice/stack_2:output:0*
Index0*
T0*!
_output_shapes
: *

begin_mask*
end_mask2
stream/strided_slicej
stream/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
stream/concat/axisธ
stream/concatConcatV2stream/strided_slice:output:0gru_1/ExpandDims:output:0stream/concat/axis:output:0*
N*
T0*#
_output_shapes
:2
stream/concatศ
stream/AssignVariableOpAssignVariableOp-stream_readvariableop_streaming_stream_statesstream/concat:output:0^stream/ReadVariableOp*
_output_shapes
 *
dtype02
stream/AssignVariableOp
stream/flatten/ConstConst^stream/AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2
stream/flatten/Const
stream/flatten/ReshapeReshapestream/concat:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshape{
dropout/IdentityIdentitystream/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identityฏ
dense/MatMul/ReadVariableOpReadVariableOp2dense_matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru_1/AssignVariableOp^stream/AssignVariableOp*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::20
gru_1/AssignVariableOpgru_1/AssignVariableOp22
stream/AssignVariableOpstream/AssignVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
฿&
โ
?__inference_gru_1_layer_call_and_return_conditional_losses_6239

inputs1
-cell_readvariableop_streaming_gru_1_cell_bias:
6cell_matmul_readvariableop_streaming_gru_1_cell_kernel6
2cell_matmul_1_readvariableop_streaming_input_stateH
Dcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel
identityขAssignVariableOpe
SqueezeSqueezeinputs*
T0*
_output_shapes

:*
squeeze_dims
2	
Squeeze
cell/ReadVariableOpReadVariableOp-cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
cell/ReadVariableOp{
cell/unstackUnpackcell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
cell/unstackฐ
cell/MatMul/ReadVariableOpReadVariableOp6cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02
cell/MatMul/ReadVariableOp
cell/MatMulMatMulSqueeze:output:0"cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
cell/MatMul
cell/BiasAddBiasAddcell/MatMul:product:0cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2

cell/splitฐ
cell/MatMul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/MatMul_1/ReadVariableOpว
cell/MatMul_1/ReadVariableOp_1ReadVariableOpDcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02 
cell/MatMul_1/ReadVariableOp_1?
cell/MatMul_1MatMul$cell/MatMul_1/ReadVariableOp:value:0&cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
cell/MatMul_1
cell/BiasAdd_1BiasAddcell/MatMul_1:product:0cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
cell/BiasAdd_1q
cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
cell/split_1s
cell/addAddV2cell/split:output:0cell/split_1:output:0*
T0*
_output_shapes
:	2

cell/add_
cell/SigmoidSigmoidcell/add:z:0*
T0*
_output_shapes
:	2
cell/Sigmoidw

cell/add_1AddV2cell/split:output:1cell/split_1:output:1*
T0*
_output_shapes
:	2

cell/add_1e
cell/Sigmoid_1Sigmoidcell/add_1:z:0*
T0*
_output_shapes
:	2
cell/Sigmoid_1p
cell/mulMulcell/Sigmoid_1:y:0cell/split_1:output:2*
T0*
_output_shapes
:	2

cell/muln

cell/add_2AddV2cell/split:output:2cell/mul:z:0*
T0*
_output_shapes
:	2

cell/add_2X
	cell/TanhTanhcell/add_2:z:0*
T0*
_output_shapes
:	2
	cell/Tanhช
cell/mul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/mul_1/ReadVariableOp~

cell/mul_1Mulcell/Sigmoid:y:0!cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

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
:	2

cell/subf

cell/mul_2Mulcell/sub:z:0cell/Tanh:y:0*
T0*
_output_shapes
:	2

cell/mul_2k

cell/add_3AddV2cell/mul_1:z:0cell/mul_2:z:0*
T0*
_output_shapes
:	2

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
:2

ExpandDimsv
IdentityIdentityExpandDims:output:0^AssignVariableOp*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:::::2$
AssignVariableOpAssignVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs

ญ
"__inference_signature_wrapper_6500
input_audio
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCallํ
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *(
f#R!
__inference__wrapped_model_61472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
"
_output_shapes
:
%
_user_specified_nameinput_audio
์

$__inference_dense_layer_call_fn_7044

inputs
streaming_dense_kernel
streaming_dense_bias
identityขStatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_63282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
ุ

`
A__inference_dropout_layer_call_and_return_conditional_losses_6300

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
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/Shapeฌ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
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
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
๖

&__inference_dense_2_layer_call_fn_7079

inputs
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_63732
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
๑
B
&__inference_dropout_layer_call_fn_7027

inputs
identityผ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_63052
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
ส%
เ
__inference__traced_save_7135
file_prefix4
0savev2_streaming_input_state_read_readvariableop6
2savev2_streaming_stream_states_read_readvariableop5
1savev2_streaming_dense_kernel_read_readvariableop3
/savev2_streaming_dense_bias_read_readvariableop7
3savev2_streaming_dense_1_kernel_read_readvariableop5
1savev2_streaming_dense_1_bias_read_readvariableop7
3savev2_streaming_dense_2_kernel_read_readvariableop5
1savev2_streaming_dense_2_bias_read_readvariableop:
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
value3B1 B+_temp_26b2bdac30f3455d8b1303394521a5f9/part2	
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
ShardedFilename๖
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
value?B๛B;layer_with_weights-0/input_state/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/states/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*+
value"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_streaming_input_state_read_readvariableop2savev2_streaming_stream_states_read_readvariableop1savev2_streaming_dense_kernel_read_readvariableop/savev2_streaming_dense_bias_read_readvariableop3savev2_streaming_dense_1_kernel_read_readvariableop1savev2_streaming_dense_1_bias_read_readvariableop3savev2_streaming_dense_2_kernel_read_readvariableop1savev2_streaming_dense_2_bias_read_readvariableop6savev2_streaming_gru_1_cell_kernel_read_readvariableop@savev2_streaming_gru_1_cell_recurrent_kernel_read_readvariableop4savev2_streaming_gru_1_cell_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
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

identity_1Identity_1:output:0*
_input_shapesy
w: :	::
::
::	::	ฐ	:
ฐ	:	ฐ	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	:)%
#
_output_shapes
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%	!

_output_shapes
:	ฐ	:&
"
 
_output_shapes
:
ฐ	:%!

_output_shapes
:	ฐ	:

_output_shapes
: 
เ!
ฤ
F__inference_functional_1_layer_call_and_return_conditional_losses_6431

inputs#
gru_1_streaming_gru_1_cell_bias%
!gru_1_streaming_gru_1_cell_kernel
gru_1_streaming_input_state/
+gru_1_streaming_gru_1_cell_recurrent_kernel"
stream_streaming_stream_states 
dense_streaming_dense_kernel
dense_streaming_dense_bias$
 dense_1_streaming_dense_1_kernel"
dense_1_streaming_dense_1_bias$
 dense_2_streaming_dense_2_kernel"
dense_2_streaming_dense_2_bias
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdense_2/StatefulPartitionedCallขdropout/StatefulPartitionedCallขgru_1/StatefulPartitionedCallขstream/StatefulPartitionedCall๙
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_streaming_gru_1_cell_bias!gru_1_streaming_gru_1_cell_kernelgru_1_streaming_input_state+gru_1_streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_62392
gru_1/StatefulPartitionedCallข
stream/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0stream_streaming_stream_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_62782 
stream/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_63002!
dropout/StatefulPartitionedCallภ
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_63282
dense/StatefulPartitionedCallฬ
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_63512!
dense_1/StatefulPartitionedCallอ
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_63732!
dense_2/StatefulPartitionedCallบ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^gru_1/StatefulPartitionedCall^stream/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs


@__inference_stream_layer_call_and_return_conditional_losses_6994

inputs*
&readvariableop_streaming_stream_states
identityขAssignVariableOp
ReadVariableOpReadVariableOp&readvariableop_streaming_stream_states*#
_output_shapes
:*
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
: *

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
:2
concatฅ
AssignVariableOpAssignVariableOp&readvariableop_streaming_stream_statesconcat:output:0^ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
flatten/ConstConst^AssignVariableOp*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Const
flatten/ReshapeReshapeconcat:output:0flatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshapew
IdentityIdentityflatten/Reshape:output:0^AssignVariableOp*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::2$
AssignVariableOpAssignVariableOp:K G
#
_output_shapes
:
 
_user_specified_nameinputs
๘

&__inference_dense_1_layer_call_fn_7062

inputs
streaming_dense_1_kernel
streaming_dense_1_bias
identityขStatefulPartitionedCall
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_63512
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
ึ
ม
?__inference_dense_layer_call_and_return_conditional_losses_7037

inputs0
,matmul_readvariableop_streaming_dense_kernel/
+biasadd_readvariableop_streaming_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp,matmul_readvariableop_streaming_dense_kernel* 
_output_shapes
:
*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
฿&
โ
?__inference_gru_1_layer_call_and_return_conditional_losses_6962

inputs1
-cell_readvariableop_streaming_gru_1_cell_bias:
6cell_matmul_readvariableop_streaming_gru_1_cell_kernel6
2cell_matmul_1_readvariableop_streaming_input_stateH
Dcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel
identityขAssignVariableOpe
SqueezeSqueezeinputs*
T0*
_output_shapes

:*
squeeze_dims
2	
Squeeze
cell/ReadVariableOpReadVariableOp-cell_readvariableop_streaming_gru_1_cell_bias*
_output_shapes
:	ฐ	*
dtype02
cell/ReadVariableOp{
cell/unstackUnpackcell/ReadVariableOp:value:0*
T0*"
_output_shapes
:ฐ	:ฐ	*	
num2
cell/unstackฐ
cell/MatMul/ReadVariableOpReadVariableOp6cell_matmul_readvariableop_streaming_gru_1_cell_kernel*
_output_shapes
:	ฐ	*
dtype02
cell/MatMul/ReadVariableOp
cell/MatMulMatMulSqueeze:output:0"cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	ฐ	2
cell/MatMul
cell/BiasAddBiasAddcell/MatMul:product:0cell/unstack:output:0*
T0*
_output_shapes
:	ฐ	2
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
!:	:	:	*
	num_split2

cell/splitฐ
cell/MatMul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/MatMul_1/ReadVariableOpว
cell/MatMul_1/ReadVariableOp_1ReadVariableOpDcell_matmul_1_readvariableop_1_streaming_gru_1_cell_recurrent_kernel* 
_output_shapes
:
ฐ	*
dtype02 
cell/MatMul_1/ReadVariableOp_1?
cell/MatMul_1MatMul$cell/MatMul_1/ReadVariableOp:value:0&cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	ฐ	2
cell/MatMul_1
cell/BiasAdd_1BiasAddcell/MatMul_1:product:0cell/unstack:output:1*
T0*
_output_shapes
:	ฐ	2
cell/BiasAdd_1q
cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ????2
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
!:	:	:	*
	num_split2
cell/split_1s
cell/addAddV2cell/split:output:0cell/split_1:output:0*
T0*
_output_shapes
:	2

cell/add_
cell/SigmoidSigmoidcell/add:z:0*
T0*
_output_shapes
:	2
cell/Sigmoidw

cell/add_1AddV2cell/split:output:1cell/split_1:output:1*
T0*
_output_shapes
:	2

cell/add_1e
cell/Sigmoid_1Sigmoidcell/add_1:z:0*
T0*
_output_shapes
:	2
cell/Sigmoid_1p
cell/mulMulcell/Sigmoid_1:y:0cell/split_1:output:2*
T0*
_output_shapes
:	2

cell/muln

cell/add_2AddV2cell/split:output:2cell/mul:z:0*
T0*
_output_shapes
:	2

cell/add_2X
	cell/TanhTanhcell/add_2:z:0*
T0*
_output_shapes
:	2
	cell/Tanhช
cell/mul_1/ReadVariableOpReadVariableOp2cell_matmul_1_readvariableop_streaming_input_state*
_output_shapes
:	*
dtype02
cell/mul_1/ReadVariableOp~

cell/mul_1Mulcell/Sigmoid:y:0!cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2

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
:	2

cell/subf

cell/mul_2Mulcell/sub:z:0cell/Tanh:y:0*
T0*
_output_shapes
:	2

cell/mul_2k

cell/add_3AddV2cell/mul_1:z:0cell/mul_2:z:0*
T0*
_output_shapes
:	2

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
:2

ExpandDimsv
IdentityIdentityExpandDims:output:0^AssignVariableOp*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:::::2$
AssignVariableOpAssignVariableOp:J F
"
_output_shapes
:
 
_user_specified_nameinputs
?
ถ
+__inference_functional_1_layer_call_fn_6687
input_audio
streaming_gru_1_cell_bias
streaming_gru_1_cell_kernel
streaming_input_state)
%streaming_gru_1_cell_recurrent_kernel
streaming_stream_states
streaming_dense_kernel
streaming_dense_bias
streaming_dense_1_kernel
streaming_dense_1_bias
streaming_dense_2_kernel
streaming_dense_2_bias
identityขStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_audiostreaming_gru_1_cell_biasstreaming_gru_1_cell_kernelstreaming_input_state%streaming_gru_1_cell_recurrent_kernelstreaming_stream_statesstreaming_dense_kernelstreaming_dense_biasstreaming_dense_1_kernelstreaming_dense_1_biasstreaming_dense_2_kernelstreaming_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_64682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:?????????:::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
+
_output_shapes
:?????????
%
_user_specified_nameinput_audio
ธ 
ข
F__inference_functional_1_layer_call_and_return_conditional_losses_6468

inputs#
gru_1_streaming_gru_1_cell_bias%
!gru_1_streaming_gru_1_cell_kernel
gru_1_streaming_input_state/
+gru_1_streaming_gru_1_cell_recurrent_kernel"
stream_streaming_stream_states 
dense_streaming_dense_kernel
dense_streaming_dense_bias$
 dense_1_streaming_dense_1_kernel"
dense_1_streaming_dense_1_bias$
 dense_2_streaming_dense_2_kernel"
dense_2_streaming_dense_2_bias
identityขdense/StatefulPartitionedCallขdense_1/StatefulPartitionedCallขdense_2/StatefulPartitionedCallขgru_1/StatefulPartitionedCallขstream/StatefulPartitionedCall๙
gru_1/StatefulPartitionedCallStatefulPartitionedCallinputsgru_1_streaming_gru_1_cell_bias!gru_1_streaming_gru_1_cell_kernelgru_1_streaming_input_state+gru_1_streaming_gru_1_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_gru_1_layer_call_and_return_conditional_losses_62392
gru_1/StatefulPartitionedCallข
stream/StatefulPartitionedCallStatefulPartitionedCall&gru_1/StatefulPartitionedCall:output:0stream_streaming_stream_states*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_62782 
stream/StatefulPartitionedCallํ
dropout/PartitionedCallPartitionedCall'stream/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_63052
dropout/PartitionedCallธ
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_63282
dense/StatefulPartitionedCallฬ
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_63512!
dense_1/StatefulPartitionedCallอ
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
*2
config_proto" 

CPU

GPU2*0,1J 8 *J
fERC
A__inference_dense_2_layer_call_and_return_conditional_losses_63732!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru_1/StatefulPartitionedCall^stream/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*M
_input_shapes<
:::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2>
gru_1/StatefulPartitionedCallgru_1/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ด
ว
A__inference_dense_1_layer_call_and_return_conditional_losses_7055

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
serving_default_input_audio:02
dense_2'
StatefulPartitionedCall:0tensorflow/serving/predict:็ฺ
ข2
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
h_default_save_signature
i__call__
*j&call_and_return_all_conditional_losses"/
_tf_keras_networkํ.{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}, "name": "input_audio", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "mode": "STREAM_INTERNAL_STATE_INFERENCE", "inference_batch_size": 1, "units": 400, "return_sequences": 0, "unroll": true, "stateful": true}, "name": "gru_1", "inbound_nodes": [[["input_audio", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 400], "ring_buffer_size_in_time_dim": 1}, "name": "stream", "inbound_nodes": [[["gru_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_audio", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 13]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}, "name": "input_audio", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "mode": "STREAM_INTERNAL_STATE_INFERENCE", "inference_batch_size": 1, "units": 400, "return_sequences": 0, "unroll": true, "stateful": true}, "name": "gru_1", "inbound_nodes": [[["input_audio", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 400], "ring_buffer_size_in_time_dim": 1}, "name": "stream", "inbound_nodes": [[["gru_1", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_audio", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
๓"๐
_tf_keras_input_layerะ{"class_name": "InputLayer", "name": "input_audio", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 13]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 1, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_audio"}}
ี
input_state
gru_cell
trainable_variables
regularization_losses
	variables
	keras_api
k__call__
*l&call_and_return_all_conditional_losses"ง
_tf_keras_layer{"class_name": "GRU", "name": "gru_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "gru_1", "trainable": true, "dtype": "float32", "mode": "STREAM_INTERNAL_STATE_INFERENCE", "inference_batch_size": 1, "units": 400, "return_sequences": 0, "unroll": true, "stateful": true}}
่
cell
state_shape

states
trainable_variables
regularization_losses
	variables
	keras_api
m__call__
*n&call_and_return_all_conditional_losses"ฒ
_tf_keras_layer{"class_name": "Stream", "name": "stream", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "STREAM_INTERNAL_STATE_INFERENCE", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 400], "ring_buffer_size_in_time_dim": 1}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 400]}}
แ
trainable_variables
regularization_losses
	variables
	keras_api
o__call__
*p&call_and_return_all_conditional_losses"า
_tf_keras_layerธ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
ง

kernel
bias
 trainable_variables
!regularization_losses
"	variables
#	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer่{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}}
ฉ

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_layer๊{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
ช

*kernel
+bias
,trainable_variables
-regularization_losses
.	variables
/	keras_api
u__call__
*v&call_and_return_all_conditional_losses"
_tf_keras_layer๋{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
_
00
11
22
3
4
$5
%6
*7
+8"
trackable_list_wrapper
 "
trackable_list_wrapper
n
00
11
22
3
4
5
6
$7
%8
*9
+10"
trackable_list_wrapper
ส
3layer_metrics

4layers
trainable_variables
5layer_regularization_losses
6metrics
7non_trainable_variables
	regularization_losses

	variables
i__call__
h_default_save_signature
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
,
wserving_default"
signature_map
&:$	2streaming/input_state


0kernel
1recurrent_kernel
2bias
8trainable_variables
9regularization_losses
:	variables
;	keras_api
x__call__
*y&call_and_return_all_conditional_losses"ฺ
_tf_keras_layerภ{"class_name": "GRUCell", "name": "cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "cell", "trainable": true, "dtype": "float32", "units": 400, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
<
00
11
22
3"
trackable_list_wrapper
ญ
<layer_metrics

=layers
trainable_variables
>layer_regularization_losses
?metrics
@non_trainable_variables
regularization_losses
	variables
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
โ
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
z__call__
*{&call_and_return_all_conditional_losses"ำ
_tf_keras_layerน{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
,:*2streaming/stream/states
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
ญ
Elayer_metrics

Flayers
trainable_variables
Glayer_regularization_losses
Hmetrics
Inon_trainable_variables
regularization_losses
	variables
m__call__
*n&call_and_return_all_conditional_losses
&n"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
Jlayer_metrics

Klayers
trainable_variables
Llayer_regularization_losses
Mmetrics
Nnon_trainable_variables
regularization_losses
	variables
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
*:(
2streaming/dense/kernel
#:!2streaming/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
ญ
Olayer_metrics

Players
 trainable_variables
Qlayer_regularization_losses
Rmetrics
Snon_trainable_variables
!regularization_losses
"	variables
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
,:*
2streaming/dense_1/kernel
%:#2streaming/dense_1/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
ญ
Tlayer_metrics

Ulayers
&trainable_variables
Vlayer_regularization_losses
Wmetrics
Xnon_trainable_variables
'regularization_losses
(	variables
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
+:)	2streaming/dense_2/kernel
$:"2streaming/dense_2/bias
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
ญ
Ylayer_metrics

Zlayers
,trainable_variables
[layer_regularization_losses
\metrics
]non_trainable_variables
-regularization_losses
.	variables
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
.:,	ฐ	2streaming/gru_1/cell/kernel
9:7
ฐ	2%streaming/gru_1/cell/recurrent_kernel
,:*	ฐ	2streaming/gru_1/cell/bias
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
 "
trackable_list_wrapper
5
00
11
22"
trackable_list_wrapper
ญ
^layer_metrics

_layers
8trainable_variables
`layer_regularization_losses
ametrics
bnon_trainable_variables
9regularization_losses
:	variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ญ
clayer_metrics

dlayers
Atrainable_variables
elayer_regularization_losses
fmetrics
gnon_trainable_variables
Bregularization_losses
C	variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
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
ๅ2โ
__inference__wrapped_model_6147พ
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
input_audio?????????
๚2๗
+__inference_functional_1_layer_call_fn_6858
+__inference_functional_1_layer_call_fn_6671
+__inference_functional_1_layer_call_fn_6874
+__inference_functional_1_layer_call_fn_6687ภ
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
ๆ2ใ
F__inference_functional_1_layer_call_and_return_conditional_losses_6655
F__inference_functional_1_layer_call_and_return_conditional_losses_6842
F__inference_functional_1_layer_call_and_return_conditional_losses_6581
F__inference_functional_1_layer_call_and_return_conditional_losses_6768ภ
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
2
$__inference_gru_1_layer_call_fn_6971
$__inference_gru_1_layer_call_fn_6980ณ
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
ป2ธ
?__inference_gru_1_layer_call_and_return_conditional_losses_6918
?__inference_gru_1_layer_call_and_return_conditional_losses_6962ณ
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
ฯ2ฬ
%__inference_stream_layer_call_fn_7000ข
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
@__inference_stream_layer_call_and_return_conditional_losses_6994ข
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
2
&__inference_dropout_layer_call_fn_7022
&__inference_dropout_layer_call_fn_7027ด
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
ภ2ฝ
A__inference_dropout_layer_call_and_return_conditional_losses_7012
A__inference_dropout_layer_call_and_return_conditional_losses_7017ด
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
ฮ2ห
$__inference_dense_layer_call_fn_7044ข
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
้2ๆ
?__inference_dense_layer_call_and_return_conditional_losses_7037ข
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
&__inference_dense_1_layer_call_fn_7062ข
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
A__inference_dense_1_layer_call_and_return_conditional_losses_7055ข
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
&__inference_dense_2_layer_call_fn_7079ข
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
A__inference_dense_2_layer_call_and_return_conditional_losses_7072ข
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
"__inference_signature_wrapper_6500input_audio
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
 
__inference__wrapped_model_6147q201$%*+8ข5
.ข+
)&
input_audio?????????
ช "(ช%
#
dense_2
dense_2
A__inference_dense_1_layer_call_and_return_conditional_losses_7055L$%'ข$
ข

inputs	
ช "ข

0	
 i
&__inference_dense_1_layer_call_fn_7062?$%'ข$
ข

inputs	
ช "	
A__inference_dense_2_layer_call_and_return_conditional_losses_7072K*+'ข$
ข

inputs	
ช "ข

0
 h
&__inference_dense_2_layer_call_fn_7079>*+'ข$
ข

inputs	
ช "
?__inference_dense_layer_call_and_return_conditional_losses_7037L'ข$
ข

inputs	
ช "ข

0	
 g
$__inference_dense_layer_call_fn_7044?'ข$
ข

inputs	
ช "	
A__inference_dropout_layer_call_and_return_conditional_losses_7012L+ข(
!ข

inputs	
p
ช "ข

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_7017L+ข(
!ข

inputs	
p 
ช "ข

0	
 i
&__inference_dropout_layer_call_fn_7022?+ข(
!ข

inputs	
p
ช "	i
&__inference_dropout_layer_call_fn_7027?+ข(
!ข

inputs	
p 
ช "	ท
F__inference_functional_1_layer_call_and_return_conditional_losses_6581m201$%*+@ข=
6ข3
)&
input_audio?????????
p

 
ช "ข

0
 ท
F__inference_functional_1_layer_call_and_return_conditional_losses_6655m201$%*+@ข=
6ข3
)&
input_audio?????????
p 

 
ช "ข

0
 ฒ
F__inference_functional_1_layer_call_and_return_conditional_losses_6768h201$%*+;ข8
1ข.
$!
inputs?????????
p

 
ช "ข

0
 ฒ
F__inference_functional_1_layer_call_and_return_conditional_losses_6842h201$%*+;ข8
1ข.
$!
inputs?????????
p 

 
ช "ข

0
 
+__inference_functional_1_layer_call_fn_6671`201$%*+@ข=
6ข3
)&
input_audio?????????
p

 
ช "
+__inference_functional_1_layer_call_fn_6687`201$%*+@ข=
6ข3
)&
input_audio?????????
p 

 
ช "
+__inference_functional_1_layer_call_fn_6858[201$%*+;ข8
1ข.
$!
inputs?????????
p

 
ช "
+__inference_functional_1_layer_call_fn_6874[201$%*+;ข8
1ข.
$!
inputs?????????
p 

 
ช "
?__inference_gru_1_layer_call_and_return_conditional_losses_6918Y201.ข+
$ข!

inputs
p
ช "!ข

0
 
?__inference_gru_1_layer_call_and_return_conditional_losses_6962Y201.ข+
$ข!

inputs
p 
ช "!ข

0
 t
$__inference_gru_1_layer_call_fn_6971L201.ข+
$ข!

inputs
p
ช "t
$__inference_gru_1_layer_call_fn_6980L201.ข+
$ข!

inputs
p 
ช "
"__inference_signature_wrapper_6500w201$%*+>ข;
ข 
4ช1
/
input_audio 
input_audio"(ช%
#
dense_2
dense_2
@__inference_stream_layer_call_and_return_conditional_losses_6994O+ข(
!ข

inputs
ช "ข

0	
 k
%__inference_stream_layer_call_fn_7000B+ข(
!ข

inputs
ช "	