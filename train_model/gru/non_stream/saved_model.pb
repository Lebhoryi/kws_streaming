
Ñ£
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
¾
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878¿
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
y
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_2/kernel
r
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes
:	*
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0

gru/cell/gru_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	°	*)
shared_namegru/cell/gru_cell/kernel

,gru/cell/gru_cell/kernel/Read/ReadVariableOpReadVariableOpgru/cell/gru_cell/kernel*
_output_shapes
:	°	*
dtype0
¢
"gru/cell/gru_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
°	*3
shared_name$"gru/cell/gru_cell/recurrent_kernel

6gru/cell/gru_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOp"gru/cell/gru_cell/recurrent_kernel* 
_output_shapes
:
°	*
dtype0

gru/cell/gru_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	°	*'
shared_namegru/cell/gru_cell/bias

*gru/cell/gru_cell/bias/Read/ReadVariableOpReadVariableOpgru/cell/gru_cell/bias*
_output_shapes
:	°	*
dtype0

gru/cell/VariableVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*"
shared_namegru/cell/Variable
x
%gru/cell/Variable/Read/ReadVariableOpReadVariableOpgru/cell/Variable*
_output_shapes
:	*
dtype0

NoOpNoOp
$
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¾#
value´#B±# Bª#
¥
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
 
[
gru
trainable_variables
regularization_losses
	variables
	keras_api
m
cell
state_shape
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
h

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
h

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
?
.0
/1
02
3
4
"5
#6
(7
)8
 
?
.0
/1
02
3
4
"5
#6
(7
)8
­
1layer_metrics

2layers
trainable_variables
3layer_regularization_losses
4metrics
5non_trainable_variables
	regularization_losses

	variables
 
l
6cell
7
state_spec
8trainable_variables
9regularization_losses
:	variables
;	keras_api

.0
/1
02
 

.0
/1
02
­
<layer_metrics

=layers
trainable_variables
>layer_regularization_losses
?metrics
@non_trainable_variables
regularization_losses
	variables
R
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
 
 
 
 
­
Elayer_metrics

Flayers
trainable_variables
Glayer_regularization_losses
Hmetrics
Inon_trainable_variables
regularization_losses
	variables
 
 
 
­
Jlayer_metrics

Klayers
trainable_variables
Llayer_regularization_losses
Mmetrics
Nnon_trainable_variables
regularization_losses
	variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
Olayer_metrics

Players
trainable_variables
Qlayer_regularization_losses
Rmetrics
Snon_trainable_variables
regularization_losses
 	variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1
 

"0
#1
­
Tlayer_metrics

Ulayers
$trainable_variables
Vlayer_regularization_losses
Wmetrics
Xnon_trainable_variables
%regularization_losses
&	variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1
 

(0
)1
­
Ylayer_metrics

Zlayers
*trainable_variables
[layer_regularization_losses
\metrics
]non_trainable_variables
+regularization_losses
,	variables
^\
VARIABLE_VALUEgru/cell/gru_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE"gru/cell/gru_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEgru/cell/gru_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
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
 
~

.kernel
/recurrent_kernel
0bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
 

.0
/1
02
 

.0
/1
02
¹
blayer_metrics

clayers
8trainable_variables
dlayer_regularization_losses
emetrics
fnon_trainable_variables
9regularization_losses

gstates
:	variables
 

0
 
 
 
 
 
 
­
hlayer_metrics

ilayers
Atrainable_variables
jlayer_regularization_losses
kmetrics
lnon_trainable_variables
Bregularization_losses
C	variables
 

0
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

.0
/1
02
 

.0
/1
02
­
mlayer_metrics

nlayers
^trainable_variables
olayer_regularization_losses
pmetrics
qnon_trainable_variables
_regularization_losses
`	variables
 

60
 
 
 

r0
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
mk
VARIABLE_VALUEgru/cell/VariableFlayer_with_weights-0/gru/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUE
p
serving_default_input_1Placeholder*"
_output_shapes
:1*
dtype0*
shape:1
þ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1gru/cell/gru_cell/biasgru/cell/gru_cell/kernelgru/cell/Variable"gru/cell/gru_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *+
f&R$
"__inference_signature_wrapper_3372
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¯
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp,gru/cell/gru_cell/kernel/Read/ReadVariableOp6gru/cell/gru_cell/recurrent_kernel/Read/ReadVariableOp*gru/cell/gru_cell/bias/Read/ReadVariableOp%gru/cell/Variable/Read/ReadVariableOpConst*
Tin
2*
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
__inference__traced_save_5221
â
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasgru/cell/gru_cell/kernel"gru/cell/gru_cell/recurrent_kernelgru/cell/gru_cell/biasgru/cell/Variable*
Tin
2*
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
 __inference__traced_restore_5261Ä


B__inference_gru_cell_layer_call_and_return_conditional_losses_2290

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitz
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
MatMul_1/ReadVariableOpµ
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	°	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
Tanht
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
mul_1/ReadVariableOpj
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
::::::F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates

³
A__inference_dense_1_layer_call_and_return_conditional_losses_3229

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
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
¾
\
@__inference_stream_layer_call_and_return_conditional_losses_3158

inputs
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten/Constw
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshaped
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*"
_input_shapes
::K G
#
_output_shapes
:
 
_user_specified_nameinputs
½w
Ë
F__inference_functional_1_layer_call_and_return_conditional_losses_4112

inputs;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity¢gru/cell/AssignVariableOp¢gru/cell/while
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposeinputs gru/cell/transpose/perm:output:0*
T0*"
_output_shapes
:12
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$gru/cell/TensorArrayV2/element_shapeÔ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ñ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2©
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp¢
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
gru/cell/gru_cell/unstackÔ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOp¼
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/gru_cell/split/split_dimÜ
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitÓ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpë
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Ô
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul_1¹
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1¤
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul¢
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhÍ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3¡
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2(
&gru/cell/TensorArrayV2_1/element_shapeÚ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time¯
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterË
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_3998*$
condR
gru_cell_while_cond_3997*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileÇ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2È
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permÁ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims}
stream/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream/flatten/Const
stream/flatten/ReshapeReshapegru/ExpandDims:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshape{
dropout/IdentityIdentitystream/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity¥
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul¡
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
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
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
²
À
%functional_1_gru_cell_while_cond_2113H
Dfunctional_1_gru_cell_while_functional_1_gru_cell_while_loop_counterN
Jfunctional_1_gru_cell_while_functional_1_gru_cell_while_maximum_iterations+
'functional_1_gru_cell_while_placeholder-
)functional_1_gru_cell_while_placeholder_1-
)functional_1_gru_cell_while_placeholder_2H
Dfunctional_1_gru_cell_while_less_functional_1_gru_cell_strided_slice^
Zfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_2113___redundant_placeholder0^
Zfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_2113___redundant_placeholder1^
Zfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_2113___redundant_placeholder2^
Zfunctional_1_gru_cell_while_functional_1_gru_cell_while_cond_2113___redundant_placeholder3(
$functional_1_gru_cell_while_identity
Ü
 functional_1/gru/cell/while/LessLess'functional_1_gru_cell_while_placeholderDfunctional_1_gru_cell_while_less_functional_1_gru_cell_strided_slice*
T0*
_output_shapes
: 2"
 functional_1/gru/cell/while/Less
$functional_1/gru/cell/while/IdentityIdentity$functional_1/gru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2&
$functional_1/gru/cell/while/Identity"U
$functional_1_gru_cell_while_identity-functional_1/gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ÿ
Ô
"__inference_gru_layer_call_fn_4464

inputs
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kernel*
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
GPU2*0,1J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_31272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:1::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:1
 
_user_specified_nameinputs


B__inference_gru_cell_layer_call_and_return_conditional_losses_2332

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitz
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
MatMul_1/ReadVariableOpµ
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1MatMulMatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	°	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
Tanht
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:	*
dtype02
mul_1/ReadVariableOpj
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
::::::F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
öZ

=__inference_gru_layer_call_and_return_conditional_losses_4446

inputs7
3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@
<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel;
7cell_gru_cell_matmul_1_readvariableop_gru_cell_variableN
Jcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity¢cell/AssignVariableOp¢
cell/while
cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose/perm
cell/transpose	Transposeinputscell/transpose/perm:output:0*
T0*"
_output_shapes
:12
cell/transposem

cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2

cell/Shape~
cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice/stack
cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_1
cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_2
cell/strided_sliceStridedSlicecell/Shape:output:0!cell/strided_slice/stack:output:0#cell/strided_slice/stack_1:output:0#cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cell/strided_slice
 cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 cell/TensorArrayV2/element_shapeÄ
cell/TensorArrayV2TensorListReserve)cell/TensorArrayV2/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2É
:cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:cell/TensorArrayUnstack/TensorListFromTensor/element_shape
,cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcell/transpose:y:0Ccell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,cell/TensorArrayUnstack/TensorListFromTensor
cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_1/stack
cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_1
cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_2
cell/strided_slice_1StridedSlicecell/transpose:y:0#cell/strided_slice_1/stack:output:0%cell/strided_slice_1/stack_1:output:0%cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
cell/strided_slice_1±
cell/gru_cell/ReadVariableOpReadVariableOp3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
cell/gru_cell/ReadVariableOp
cell/gru_cell/unstackUnpack$cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
cell/gru_cell/unstackÈ
#cell/gru_cell/MatMul/ReadVariableOpReadVariableOp<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02%
#cell/gru_cell/MatMul/ReadVariableOp¬
cell/gru_cell/MatMulMatMulcell/strided_slice_1:output:0+cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/MatMul£
cell/gru_cell/BiasAddBiasAddcell/gru_cell/MatMul:product:0cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/BiasAddl
cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/gru_cell/Const
cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
cell/gru_cell/split/split_dimÌ
cell/gru_cell/splitSplit&cell/gru_cell/split/split_dim:output:0cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/splitÇ
%cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02'
%cell/gru_cell/MatMul_1/ReadVariableOpß
'cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02)
'cell/gru_cell/MatMul_1/ReadVariableOp_1Ä
cell/gru_cell/MatMul_1MatMul-cell/gru_cell/MatMul_1/ReadVariableOp:value:0/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/MatMul_1©
cell/gru_cell/BiasAdd_1BiasAdd cell/gru_cell/MatMul_1:product:0cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
cell/gru_cell/BiasAdd_1
cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
cell/gru_cell/Const_1
cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
cell/gru_cell/split_1/split_dim
cell/gru_cell/split_1SplitV cell/gru_cell/BiasAdd_1:output:0cell/gru_cell/Const_1:output:0(cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/split_1
cell/gru_cell/addAddV2cell/gru_cell/split:output:0cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/addz
cell/gru_cell/SigmoidSigmoidcell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid
cell/gru_cell/add_1AddV2cell/gru_cell/split:output:1cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/add_1
cell/gru_cell/Sigmoid_1Sigmoidcell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid_1
cell/gru_cell/mulMulcell/gru_cell/Sigmoid_1:y:0cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/gru_cell/mul
cell/gru_cell/add_2AddV2cell/gru_cell/split:output:2cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_2s
cell/gru_cell/TanhTanhcell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/TanhÁ
"cell/gru_cell/mul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02$
"cell/gru_cell/mul_1/ReadVariableOp¢
cell/gru_cell/mul_1Mulcell/gru_cell/Sigmoid:y:0*cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_1o
cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/gru_cell/sub/x
cell/gru_cell/subSubcell/gru_cell/sub/x:output:0cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/sub
cell/gru_cell/mul_2Mulcell/gru_cell/sub:z:0cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_2
cell/gru_cell/add_3AddV2cell/gru_cell/mul_1:z:0cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_3
"cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2$
"cell/TensorArrayV2_1/element_shapeÊ
cell/TensorArrayV2_1TensorListReserve+cell/TensorArrayV2_1/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2_1X
	cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	cell/time£
cell/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
cell/ReadVariableOp
cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
cell/while/maximum_iterationst
cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cell/while/loop_counter

cell/whileWhile cell/while/loop_counter:output:0&cell/while/maximum_iterations:output:0cell/time:output:0cell/TensorArrayV2_1:handle:0cell/ReadVariableOp:value:0cell/strided_slice:output:0<cell/TensorArrayUnstack/TensorListFromTensor:output_handle:03cell_gru_cell_readvariableop_gru_cell_gru_cell_bias<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	* 
bodyR
cell_while_body_4354* 
condR
cell_while_cond_4353*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2

cell/while¿
5cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     27
5cell/TensorArrayV2Stack/TensorListStack/element_shapeô
'cell/TensorArrayV2Stack/TensorListStackTensorListStackcell/while:output:3>cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02)
'cell/TensorArrayV2Stack/TensorListStack
cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
cell/strided_slice_2/stack
cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_2/stack_1
cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_2/stack_2°
cell/strided_slice_2StridedSlice0cell/TensorArrayV2Stack/TensorListStack:tensor:0#cell/strided_slice_2/stack:output:0%cell/strided_slice_2/stack_1:output:0%cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
cell/strided_slice_2
cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose_1/perm±
cell/transpose_1	Transpose0cell/TensorArrayV2Stack/TensorListStack:tensor:0cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
cell/transpose_1p
cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
cell/runtime
cell/AssignVariableOpAssignVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variablecell/while:output:4^cell/ReadVariableOp&^cell/gru_cell/MatMul_1/ReadVariableOp#^cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
cell/AssignVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimscell/strided_slice_2:output:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDims
IdentityIdentityExpandDims:output:0^cell/AssignVariableOp^cell/while*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:1::::2.
cell/AssignVariableOpcell/AssignVariableOp2

cell/while
cell/while:J F
"
_output_shapes
:1
 
_user_specified_nameinputs
³
³
A__inference_dense_2_layer_call_and_return_conditional_losses_4547

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
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
þ

B__inference_gru_cell_layer_call_and_return_conditional_losses_2428

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split¯
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	°	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!::	::::F B

_output_shapes

:
 
_user_specified_nameinputs:GC

_output_shapes
:	
 
_user_specified_namestates
öZ

=__inference_gru_layer_call_and_return_conditional_losses_4294

inputs7
3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@
<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel;
7cell_gru_cell_matmul_1_readvariableop_gru_cell_variableN
Jcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity¢cell/AssignVariableOp¢
cell/while
cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose/perm
cell/transpose	Transposeinputscell/transpose/perm:output:0*
T0*"
_output_shapes
:12
cell/transposem

cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2

cell/Shape~
cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice/stack
cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_1
cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_2
cell/strided_sliceStridedSlicecell/Shape:output:0!cell/strided_slice/stack:output:0#cell/strided_slice/stack_1:output:0#cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cell/strided_slice
 cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 cell/TensorArrayV2/element_shapeÄ
cell/TensorArrayV2TensorListReserve)cell/TensorArrayV2/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2É
:cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:cell/TensorArrayUnstack/TensorListFromTensor/element_shape
,cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcell/transpose:y:0Ccell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,cell/TensorArrayUnstack/TensorListFromTensor
cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_1/stack
cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_1
cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_2
cell/strided_slice_1StridedSlicecell/transpose:y:0#cell/strided_slice_1/stack:output:0%cell/strided_slice_1/stack_1:output:0%cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
cell/strided_slice_1±
cell/gru_cell/ReadVariableOpReadVariableOp3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
cell/gru_cell/ReadVariableOp
cell/gru_cell/unstackUnpack$cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
cell/gru_cell/unstackÈ
#cell/gru_cell/MatMul/ReadVariableOpReadVariableOp<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02%
#cell/gru_cell/MatMul/ReadVariableOp¬
cell/gru_cell/MatMulMatMulcell/strided_slice_1:output:0+cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/MatMul£
cell/gru_cell/BiasAddBiasAddcell/gru_cell/MatMul:product:0cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/BiasAddl
cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/gru_cell/Const
cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
cell/gru_cell/split/split_dimÌ
cell/gru_cell/splitSplit&cell/gru_cell/split/split_dim:output:0cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/splitÇ
%cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02'
%cell/gru_cell/MatMul_1/ReadVariableOpß
'cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02)
'cell/gru_cell/MatMul_1/ReadVariableOp_1Ä
cell/gru_cell/MatMul_1MatMul-cell/gru_cell/MatMul_1/ReadVariableOp:value:0/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/MatMul_1©
cell/gru_cell/BiasAdd_1BiasAdd cell/gru_cell/MatMul_1:product:0cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
cell/gru_cell/BiasAdd_1
cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
cell/gru_cell/Const_1
cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
cell/gru_cell/split_1/split_dim
cell/gru_cell/split_1SplitV cell/gru_cell/BiasAdd_1:output:0cell/gru_cell/Const_1:output:0(cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/split_1
cell/gru_cell/addAddV2cell/gru_cell/split:output:0cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/addz
cell/gru_cell/SigmoidSigmoidcell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid
cell/gru_cell/add_1AddV2cell/gru_cell/split:output:1cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/add_1
cell/gru_cell/Sigmoid_1Sigmoidcell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid_1
cell/gru_cell/mulMulcell/gru_cell/Sigmoid_1:y:0cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/gru_cell/mul
cell/gru_cell/add_2AddV2cell/gru_cell/split:output:2cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_2s
cell/gru_cell/TanhTanhcell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/TanhÁ
"cell/gru_cell/mul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02$
"cell/gru_cell/mul_1/ReadVariableOp¢
cell/gru_cell/mul_1Mulcell/gru_cell/Sigmoid:y:0*cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_1o
cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/gru_cell/sub/x
cell/gru_cell/subSubcell/gru_cell/sub/x:output:0cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/sub
cell/gru_cell/mul_2Mulcell/gru_cell/sub:z:0cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_2
cell/gru_cell/add_3AddV2cell/gru_cell/mul_1:z:0cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_3
"cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2$
"cell/TensorArrayV2_1/element_shapeÊ
cell/TensorArrayV2_1TensorListReserve+cell/TensorArrayV2_1/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2_1X
	cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	cell/time£
cell/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
cell/ReadVariableOp
cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
cell/while/maximum_iterationst
cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cell/while/loop_counter

cell/whileWhile cell/while/loop_counter:output:0&cell/while/maximum_iterations:output:0cell/time:output:0cell/TensorArrayV2_1:handle:0cell/ReadVariableOp:value:0cell/strided_slice:output:0<cell/TensorArrayUnstack/TensorListFromTensor:output_handle:03cell_gru_cell_readvariableop_gru_cell_gru_cell_bias<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	* 
bodyR
cell_while_body_4202* 
condR
cell_while_cond_4201*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2

cell/while¿
5cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     27
5cell/TensorArrayV2Stack/TensorListStack/element_shapeô
'cell/TensorArrayV2Stack/TensorListStackTensorListStackcell/while:output:3>cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02)
'cell/TensorArrayV2Stack/TensorListStack
cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
cell/strided_slice_2/stack
cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_2/stack_1
cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_2/stack_2°
cell/strided_slice_2StridedSlice0cell/TensorArrayV2Stack/TensorListStack:tensor:0#cell/strided_slice_2/stack:output:0%cell/strided_slice_2/stack_1:output:0%cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
cell/strided_slice_2
cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose_1/perm±
cell/transpose_1	Transpose0cell/TensorArrayV2Stack/TensorListStack:tensor:0cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
cell/transpose_1p
cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
cell/runtime
cell/AssignVariableOpAssignVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variablecell/while:output:4^cell/ReadVariableOp&^cell/gru_cell/MatMul_1/ReadVariableOp#^cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
cell/AssignVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimscell/strided_slice_2:output:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDims
IdentityIdentityExpandDims:output:0^cell/AssignVariableOp^cell/while*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:1::::2.
cell/AssignVariableOpcell/AssignVariableOp2

cell/while
cell/while:J F
"
_output_shapes
:1
 
_user_specified_nameinputs
Á	
à
'__inference_gru_cell_layer_call_fn_5066

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_50582
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0


Ë
+__inference_functional_1_layer_call_fn_4142

inputs
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_33422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ1::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

 
B__inference_gru_cell_layer_call_and_return_conditional_losses_4915

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitu
MatMul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOpµ
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanho
mul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
::::::F B

_output_shapes

:
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0


B__inference_gru_cell_layer_call_and_return_conditional_losses_5106

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split¯
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOps
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	°	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhV
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!::	::::F B

_output_shapes

:
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
¡K
	
gru_cell_while_body_3817.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÕ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemÑ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp´
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2!
gru/cell/while/gru_cell/unstackè
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpæ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2 
gru/cell/while/gru_cell/MatMulË
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'gru/cell/while/gru_cell/split/split_dimô
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitù
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpÏ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2"
 gru/cell/while/gru_cell/MatMul_1Ñ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2!
gru/cell/while/gru_cell/Const_1¡
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1¿
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidÃ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1¼
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulº
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/x¸
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ª
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Ì
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
¸4
¸
>__inference_cell_layer_call_and_return_conditional_losses_2812

inputs
gru_cell_gru_cell_variable#
gru_cell_gru_cell_gru_cell_bias%
!gru_cell_gru_cell_gru_cell_kernel/
+gru_cell_gru_cell_gru_cell_recurrent_kernel
identity¢AssignVariableOp¢ gru_cell/StatefulPartitionedCall¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ó
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_1
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_gru_cell_variablegru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_23322"
 gru_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time|
ReadVariableOpReadVariableOpgru_cell_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_2751*
condR
while_cond_2750*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
AssignVariableOpAssignVariableOpgru_cell_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 *
dtype02
AssignVariableOp¢
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2$
AssignVariableOpAssignVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


à
'__inference_gru_cell_layer_call_fn_5168

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_24682
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!::	:::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0

×
#__inference_cell_layer_call_fn_4863
inputs_0
gru_cell_variable
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_cell_variablegru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *G
fBR@
>__inference_cell_layer_call_and_return_conditional_losses_27132
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¤
_
A__inference_dropout_layer_call_and_return_conditional_losses_4492

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
®
­
?__inference_dense_layer_call_and_return_conditional_losses_3206

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
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
Î

&__inference_dense_2_layer_call_fn_4554

inputs
dense_2_kernel
dense_2_bias
identity¢StatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCallinputsdense_2_kerneldense_2_bias*
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
A__inference_dense_2_layer_call_and_return_conditional_losses_32512
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
öZ

=__inference_gru_layer_call_and_return_conditional_losses_3127

inputs7
3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@
<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel;
7cell_gru_cell_matmul_1_readvariableop_gru_cell_variableN
Jcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity¢cell/AssignVariableOp¢
cell/while
cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose/perm
cell/transpose	Transposeinputscell/transpose/perm:output:0*
T0*"
_output_shapes
:12
cell/transposem

cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2

cell/Shape~
cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice/stack
cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_1
cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice/stack_2
cell/strided_sliceStridedSlicecell/Shape:output:0!cell/strided_slice/stack:output:0#cell/strided_slice/stack_1:output:0#cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cell/strided_slice
 cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 cell/TensorArrayV2/element_shapeÄ
cell/TensorArrayV2TensorListReserve)cell/TensorArrayV2/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2É
:cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2<
:cell/TensorArrayUnstack/TensorListFromTensor/element_shape
,cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorcell/transpose:y:0Ccell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,cell/TensorArrayUnstack/TensorListFromTensor
cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_1/stack
cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_1
cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_1/stack_2
cell/strided_slice_1StridedSlicecell/transpose:y:0#cell/strided_slice_1/stack:output:0%cell/strided_slice_1/stack_1:output:0%cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
cell/strided_slice_1±
cell/gru_cell/ReadVariableOpReadVariableOp3cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
cell/gru_cell/ReadVariableOp
cell/gru_cell/unstackUnpack$cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
cell/gru_cell/unstackÈ
#cell/gru_cell/MatMul/ReadVariableOpReadVariableOp<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02%
#cell/gru_cell/MatMul/ReadVariableOp¬
cell/gru_cell/MatMulMatMulcell/strided_slice_1:output:0+cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/MatMul£
cell/gru_cell/BiasAddBiasAddcell/gru_cell/MatMul:product:0cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/BiasAddl
cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/gru_cell/Const
cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
cell/gru_cell/split/split_dimÌ
cell/gru_cell/splitSplit&cell/gru_cell/split/split_dim:output:0cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/splitÇ
%cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02'
%cell/gru_cell/MatMul_1/ReadVariableOpß
'cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02)
'cell/gru_cell/MatMul_1/ReadVariableOp_1Ä
cell/gru_cell/MatMul_1MatMul-cell/gru_cell/MatMul_1/ReadVariableOp:value:0/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
cell/gru_cell/MatMul_1©
cell/gru_cell/BiasAdd_1BiasAdd cell/gru_cell/MatMul_1:product:0cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
cell/gru_cell/BiasAdd_1
cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
cell/gru_cell/Const_1
cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2!
cell/gru_cell/split_1/split_dim
cell/gru_cell/split_1SplitV cell/gru_cell/BiasAdd_1:output:0cell/gru_cell/Const_1:output:0(cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/gru_cell/split_1
cell/gru_cell/addAddV2cell/gru_cell/split:output:0cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/gru_cell/addz
cell/gru_cell/SigmoidSigmoidcell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid
cell/gru_cell/add_1AddV2cell/gru_cell/split:output:1cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/gru_cell/add_1
cell/gru_cell/Sigmoid_1Sigmoidcell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/Sigmoid_1
cell/gru_cell/mulMulcell/gru_cell/Sigmoid_1:y:0cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/gru_cell/mul
cell/gru_cell/add_2AddV2cell/gru_cell/split:output:2cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_2s
cell/gru_cell/TanhTanhcell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/TanhÁ
"cell/gru_cell/mul_1/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02$
"cell/gru_cell/mul_1/ReadVariableOp¢
cell/gru_cell/mul_1Mulcell/gru_cell/Sigmoid:y:0*cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_1o
cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/gru_cell/sub/x
cell/gru_cell/subSubcell/gru_cell/sub/x:output:0cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/sub
cell/gru_cell/mul_2Mulcell/gru_cell/sub:z:0cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/gru_cell/mul_2
cell/gru_cell/add_3AddV2cell/gru_cell/mul_1:z:0cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/gru_cell/add_3
"cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2$
"cell/TensorArrayV2_1/element_shapeÊ
cell/TensorArrayV2_1TensorListReserve+cell/TensorArrayV2_1/element_shape:output:0cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
cell/TensorArrayV2_1X
	cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	cell/time£
cell/ReadVariableOpReadVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
cell/ReadVariableOp
cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
cell/while/maximum_iterationst
cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
cell/while/loop_counter

cell/whileWhile cell/while/loop_counter:output:0&cell/while/maximum_iterations:output:0cell/time:output:0cell/TensorArrayV2_1:handle:0cell/ReadVariableOp:value:0cell/strided_slice:output:0<cell/TensorArrayUnstack/TensorListFromTensor:output_handle:03cell_gru_cell_readvariableop_gru_cell_gru_cell_bias<cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelJcell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	* 
bodyR
cell_while_body_3035* 
condR
cell_while_cond_3034*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2

cell/while¿
5cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     27
5cell/TensorArrayV2Stack/TensorListStack/element_shapeô
'cell/TensorArrayV2Stack/TensorListStackTensorListStackcell/while:output:3>cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02)
'cell/TensorArrayV2Stack/TensorListStack
cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
cell/strided_slice_2/stack
cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cell/strided_slice_2/stack_1
cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cell/strided_slice_2/stack_2°
cell/strided_slice_2StridedSlice0cell/TensorArrayV2Stack/TensorListStack:tensor:0#cell/strided_slice_2/stack:output:0%cell/strided_slice_2/stack_1:output:0%cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
cell/strided_slice_2
cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
cell/transpose_1/perm±
cell/transpose_1	Transpose0cell/TensorArrayV2Stack/TensorListStack:tensor:0cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
cell/transpose_1p
cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
cell/runtime
cell/AssignVariableOpAssignVariableOp7cell_gru_cell_matmul_1_readvariableop_gru_cell_variablecell/while:output:4^cell/ReadVariableOp&^cell/gru_cell/MatMul_1/ReadVariableOp#^cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
cell/AssignVariableOpb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimscell/strided_slice_2:output:0ExpandDims/dim:output:0*
T0*#
_output_shapes
:2

ExpandDims
IdentityIdentityExpandDims:output:0^cell/AssignVariableOp^cell/while*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:1::::2.
cell/AssignVariableOpcell/AssignVariableOp2

cell/while
cell/while:J F
"
_output_shapes
:1
 
_user_specified_nameinputs


F__inference_functional_1_layer_call_and_return_conditional_losses_3342

inputs
gru_gru_cell_gru_cell_bias 
gru_gru_cell_gru_cell_kernel
gru_gru_cell_variable*
&gru_gru_cell_gru_cell_recurrent_kernel
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢gru/StatefulPartitionedCallÞ
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_gru_cell_gru_cell_biasgru_gru_cell_gru_cell_kernelgru_gru_cell_variable&gru_gru_cell_gru_cell_recurrent_kernel*
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
GPU2*0,1J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_31272
gru/StatefulPartitionedCallç
stream/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_31582
stream/PartitionedCallå
dropout/PartitionedCallPartitionedCallstream/PartitionedCall:output:0*
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
A__inference_dropout_layer_call_and_return_conditional_losses_31832
dropout/PartitionedCall¤
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
?__inference_dense_layer_call_and_return_conditional_losses_32062
dense/StatefulPartitionedCall¸
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
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
A__inference_dense_1_layer_call_and_return_conditional_losses_32292!
dense_1/StatefulPartitionedCall¹
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
A__inference_dense_2_layer_call_and_return_conditional_losses_32512!
dense_2/StatefulPartitionedCallõ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
¡K
	
gru_cell_while_body_3998.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÕ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemÑ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp´
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2!
gru/cell/while/gru_cell/unstackè
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpæ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2 
gru/cell/while/gru_cell/MatMulË
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'gru/cell/while/gru_cell/split/split_dimô
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitù
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpÏ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2"
 gru/cell/while/gru_cell/MatMul_1Ñ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2!
gru/cell/while/gru_cell/Const_1¡
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1¿
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidÃ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1¼
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulº
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/x¸
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ª
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Ì
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 


Ì
+__inference_functional_1_layer_call_fn_3757
input_1
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_1gru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_33422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ1::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!
_user_specified_name	input_1
ü

B__inference_gru_cell_layer_call_and_return_conditional_losses_5058

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splits
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOpµ
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanhm
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
::::::F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates


Ì
+__inference_functional_1_layer_call_fn_3742
input_1
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity¢StatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_1gru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_33072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ1::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!
_user_specified_name	input_1
#
¼
while_body_2751
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
'while_gru_cell_gru_cell_gru_cell_bias_0-
)while_gru_cell_gru_cell_gru_cell_kernel_07
3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
%while_gru_cell_gru_cell_gru_cell_bias+
'while_gru_cell_gru_cell_gru_cell_kernel5
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel¢&while/gru_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÊ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÐ
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2'while_gru_cell_gru_cell_gru_cell_bias_0)while_gru_cell_gru_cell_gru_cell_kernel_03while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_24682(
&while/gru_cell/StatefulPartitionedCalló
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¶
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3´
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
:	2
while/Identity_4"P
%while_gru_cell_gru_cell_gru_cell_bias'while_gru_cell_gru_cell_gru_cell_bias_0"T
'while_gru_cell_gru_cell_gru_cell_kernel)while_gru_cell_gru_cell_gru_cell_kernel_0"h
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
²

while_cond_4613
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_4613___redundant_placeholder02
.while_while_cond_4613___redundant_placeholder12
.while_while_cond_4613___redundant_placeholder22
.while_while_cond_4613___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ïP
î
>__inference_cell_layer_call_and_return_conditional_losses_4704
inputs_02
.gru_cell_readvariableop_gru_cell_gru_cell_bias;
7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel6
2gru_cell_matmul_1_readvariableop_gru_cell_variableI
Egru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity¢AssignVariableOp¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ó
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_1¢
gru_cell/ReadVariableOpReadVariableOp.gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
gru_cell/unstack¹
gru_cell/MatMul/ReadVariableOpReadVariableOp7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dim¸
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/split¸
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02"
 gru_cell/MatMul_1/ReadVariableOpÐ
"gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02$
"gru_cell/MatMul_1/ReadVariableOp_1°
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0*gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dimè
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_cell/Tanh²
gru_cell/mul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru_cell/mul_1/ReadVariableOp
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÍ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.gru_cell_readvariableop_gru_cell_gru_cell_bias7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_4614*
condR
while_cond_4613*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeó
AssignVariableOpAssignVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2$
AssignVariableOpAssignVariableOp2
whilewhile:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÉE

cell_while_body_4354&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2#
cell_while_cell_strided_slice_0a
]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0?
;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0H
Dcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0T
Pcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
cell_while_identity
cell_while_identity_1
cell_while_identity_2
cell_while_identity_3
cell_while_identity_4!
cell_while_cell_strided_slice_
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor=
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasF
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelR
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÍ
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2>
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeè
.cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0cell_while_placeholderEcell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype020
.cell/while/TensorArrayV2Read/TensorListGetItemÅ
"cell/while/gru_cell/ReadVariableOpReadVariableOp;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02$
"cell/while/gru_cell/ReadVariableOp¨
cell/while/gru_cell/unstackUnpack*cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
cell/while/gru_cell/unstackÜ
)cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02+
)cell/while/gru_cell/MatMul/ReadVariableOpÖ
cell/while/gru_cell/MatMulMatMul5cell/while/TensorArrayV2Read/TensorListGetItem:item:01cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/MatMul»
cell/while/gru_cell/BiasAddBiasAdd$cell/while/gru_cell/MatMul:product:0$cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/BiasAddx
cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/gru_cell/Const
#cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#cell/while/gru_cell/split/split_dimä
cell/while/gru_cell/splitSplit,cell/while/gru_cell/split/split_dim:output:0$cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/splití
+cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype02-
+cell/while/gru_cell/MatMul_1/ReadVariableOp¿
cell/while/gru_cell/MatMul_1MatMulcell_while_placeholder_23cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/MatMul_1Á
cell/while/gru_cell/BiasAdd_1BiasAdd&cell/while/gru_cell/MatMul_1:product:0$cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/BiasAdd_1
cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
cell/while/gru_cell/Const_1
%cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%cell/while/gru_cell/split_1/split_dim
cell/while/gru_cell/split_1SplitV&cell/while/gru_cell/BiasAdd_1:output:0$cell/while/gru_cell/Const_1:output:0.cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/split_1¯
cell/while/gru_cell/addAddV2"cell/while/gru_cell/split:output:0$cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add
cell/while/gru_cell/SigmoidSigmoidcell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid³
cell/while/gru_cell/add_1AddV2"cell/while/gru_cell/split:output:1$cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_1
cell/while/gru_cell/Sigmoid_1Sigmoidcell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid_1¬
cell/while/gru_cell/mulMul!cell/while/gru_cell/Sigmoid_1:y:0$cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mulª
cell/while/gru_cell/add_2AddV2"cell/while/gru_cell/split:output:2cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_2
cell/while/gru_cell/TanhTanhcell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Tanh¢
cell/while/gru_cell/mul_1Mulcell/while/gru_cell/Sigmoid:y:0cell_while_placeholder_2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_1{
cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/while/gru_cell/sub/x¨
cell/while/gru_cell/subSub"cell/while/gru_cell/sub/x:output:0cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/sub¢
cell/while/gru_cell/mul_2Mulcell/while/gru_cell/sub:z:0cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_2§
cell/while/gru_cell/add_3AddV2cell/while/gru_cell/mul_1:z:0cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_3õ
/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcell_while_placeholder_1cell_while_placeholdercell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype021
/cell/while/TensorArrayV2Write/TensorListSetItemf
cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add/y}
cell/while/addAddV2cell_while_placeholdercell/while/add/y:output:0*
T0*
_output_shapes
: 2
cell/while/addj
cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add_1/y
cell/while/add_1AddV2"cell_while_cell_while_loop_countercell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cell/while/add_1m
cell/while/IdentityIdentitycell/while/add_1:z:0*
T0*
_output_shapes
: 2
cell/while/Identity
cell/while/Identity_1Identity(cell_while_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
cell/while/Identity_1o
cell/while/Identity_2Identitycell/while/add:z:0*
T0*
_output_shapes
: 2
cell/while/Identity_2
cell/while/Identity_3Identity?cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
cell/while/Identity_3
cell/while/Identity_4Identitycell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
cell/while/Identity_4"@
cell_while_cell_strided_slicecell_while_cell_strided_slice_0"¢
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"x
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"3
cell_while_identitycell/while/Identity:output:0"7
cell_while_identity_1cell/while/Identity_1:output:0"7
cell_while_identity_2cell/while/Identity_2:output:0"7
cell_while_identity_3cell/while/Identity_3:output:0"7
cell_while_identity_4cell/while/Identity_4:output:0"¼
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
Ø

`
A__inference_dropout_layer_call_and_return_conditional_losses_3178

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
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
dropout/Shape¬
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
 *ÍÌÌ=2
dropout/GreaterEqual/y¶
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
¸	
É
gru_cell_while_cond_3431.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_3431___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_3431___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_3431___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_3431___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
¾
\
@__inference_stream_layer_call_and_return_conditional_losses_4470

inputs
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
flatten/Constw
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshaped
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*"
_input_shapes
::K G
#
_output_shapes
:
 
_user_specified_nameinputs
÷
A
%__inference_stream_layer_call_fn_4475

inputs
identity»
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
GPU2*0,1J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_31582
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*"
_input_shapes
::K G
#
_output_shapes
:
 
_user_specified_nameinputs
²

while_cond_2750
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_2750___redundant_placeholder02
.while_while_cond_2750___redundant_placeholder12
.while_while_cond_2750___redundant_placeholder22
.while_while_cond_2750___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ý
_
&__inference_dropout_layer_call_fn_4497

inputs
identity¢StatefulPartitionedCallÔ
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
A__inference_dropout_layer_call_and_return_conditional_losses_31782
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
§
Ì
F__inference_functional_1_layer_call_and_return_conditional_losses_3553
input_1;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity¢gru/cell/AssignVariableOp¢gru/cell/while
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposeinput_1 gru/cell/transpose/perm:output:0*
T0*"
_output_shapes
:12
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$gru/cell/TensorArrayV2/element_shapeÔ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ñ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2©
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp¢
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
gru/cell/gru_cell/unstackÔ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOp¼
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/gru_cell/split/split_dimÜ
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitÓ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpë
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Ô
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul_1¹
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1¤
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul¢
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhÍ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3¡
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2(
&gru/cell/TensorArrayV2_1/element_shapeÚ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time¯
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterË
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_3432*$
condR
gru_cell_while_cond_3431*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileÇ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2È
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permÁ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims}
stream/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream/flatten/Const
stream/flatten/ReshapeReshapegru/ExpandDims:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
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
dropout/dropout/ShapeÄ
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
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yÖ
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
dropout/dropout/Mul_1¥
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul¡
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
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
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!
_user_specified_name	input_1
þ

B__inference_gru_cell_layer_call_and_return_conditional_losses_2468

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split¯
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOpq
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	°	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhT
mul_1MulSigmoid:y:0states*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!::	::::F B

_output_shapes

:
 
_user_specified_nameinputs:GC

_output_shapes
:	
 
_user_specified_namestates


à
'__inference_gru_cell_layer_call_fn_5157

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1¢StatefulPartitionedCallÊ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_24282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!::	:::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
ñ
B
&__inference_dropout_layer_call_fn_4502

inputs
identity¼
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
A__inference_dropout_layer_call_and_return_conditional_losses_31832
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
¸4
¸
>__inference_cell_layer_call_and_return_conditional_losses_2713

inputs
gru_cell_gru_cell_variable#
gru_cell_gru_cell_gru_cell_bias%
!gru_cell_gru_cell_gru_cell_kernel/
+gru_cell_gru_cell_gru_cell_recurrent_kernel
identity¢AssignVariableOp¢ gru_cell/StatefulPartitionedCall¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/permz
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ó
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_1
 gru_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_1:output:0gru_cell_gru_cell_variablegru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_22902"
 gru_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time|
ReadVariableOpReadVariableOpgru_cell_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0gru_cell_gru_cell_gru_cell_bias!gru_cell_gru_cell_gru_cell_kernel+gru_cell_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_2652*
condR
while_cond_2651*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime»
AssignVariableOpAssignVariableOpgru_cell_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/StatefulPartitionedCall*
_output_shapes
 *
dtype02
AssignVariableOp¢
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp!^gru_cell/StatefulPartitionedCall^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2$
AssignVariableOpAssignVariableOp2D
 gru_cell/StatefulPartitionedCall gru_cell/StatefulPartitionedCall2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


B__inference_gru_cell_layer_call_and_return_conditional_losses_5146

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel>
:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
split¯
MatMul_1/ReadVariableOpReadVariableOp:matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOps
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2

MatMul_1q
	BiasAdd_1BiasAddMatMul_1:product:0unstack:output:1*
T0*
_output_shapes
:	°	2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim»
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2	
split_1_
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:	2
addP
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:	2	
Sigmoidc
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:	2
add_1V
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:	2
	Sigmoid_1\
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:	2
mulZ
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:	2
add_2I
TanhTanh	add_2:z:0*
T0*
_output_shapes
:	2
TanhV
mul_1MulSigmoid:y:0states_0*
T0*
_output_shapes
:	2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xX
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:	2
subR
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:	2
mul_2W
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:	2
add_3U
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:	2

IdentityY

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:	2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*4
_input_shapes#
!::	::::F B

_output_shapes

:
 
_user_specified_nameinputs:IE

_output_shapes
:	
"
_user_specified_name
states/0
Â	
Ã
"__inference_signature_wrapper_3372
input_1
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1gru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *(
f#R!
__inference__wrapped_model_22282
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:1
!
_user_specified_name	input_1
¡K
	
gru_cell_while_body_3432.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÕ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemÑ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp´
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2!
gru/cell/while/gru_cell/unstackè
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpæ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2 
gru/cell/while/gru_cell/MatMulË
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'gru/cell/while/gru_cell/split/split_dimô
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitù
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpÏ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2"
 gru/cell/while/gru_cell/MatMul_1Ñ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2!
gru/cell/while/gru_cell/Const_1¡
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1¿
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidÃ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1¼
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulº
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/x¸
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ª
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Ì
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 


Ë
+__inference_functional_1_layer_call_fn_4127

inputs
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity¢StatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCallinputsgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kerneldense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*+
_read_only_resource_inputs
		
*2
config_proto" 

CPU

GPU2*0,1J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_33072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ1::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs

 
B__inference_gru_cell_layer_call_and_return_conditional_losses_4958

inputs
states_0)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splitu
MatMul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOpµ
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanho
mul_1/ReadVariableOpReadVariableOpstates_0*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
::::::F B

_output_shapes

:
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
¡K
	
gru_cell_while_body_3613.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2+
'gru_cell_while_gru_cell_strided_slice_0i
egru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0C
?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0L
Hgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0X
Tgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
gru_cell_while_identity
gru_cell_while_identity_1
gru_cell_while_identity_2
gru_cell_while_identity_3
gru_cell_while_identity_4)
%gru_cell_while_gru_cell_strided_sliceg
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensorA
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasJ
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelV
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÕ
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2B
@gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape
2gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemegru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0gru_cell_while_placeholderIgru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype024
2gru/cell/while/TensorArrayV2Read/TensorListGetItemÑ
&gru/cell/while/gru_cell/ReadVariableOpReadVariableOp?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02(
&gru/cell/while/gru_cell/ReadVariableOp´
gru/cell/while/gru_cell/unstackUnpack.gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2!
gru/cell/while/gru_cell/unstackè
-gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02/
-gru/cell/while/gru_cell/MatMul/ReadVariableOpæ
gru/cell/while/gru_cell/MatMulMatMul9gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:05gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2 
gru/cell/while/gru_cell/MatMulË
gru/cell/while/gru_cell/BiasAddBiasAdd(gru/cell/while/gru_cell/MatMul:product:0(gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2!
gru/cell/while/gru_cell/BiasAdd
gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/gru_cell/Const
'gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2)
'gru/cell/while/gru_cell/split/split_dimô
gru/cell/while/gru_cell/splitSplit0gru/cell/while/gru_cell/split/split_dim:output:0(gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/while/gru_cell/splitù
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype021
/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpÏ
 gru/cell/while/gru_cell/MatMul_1MatMulgru_cell_while_placeholder_27gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2"
 gru/cell/while/gru_cell/MatMul_1Ñ
!gru/cell/while/gru_cell/BiasAdd_1BiasAdd*gru/cell/while/gru_cell/MatMul_1:product:0(gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2#
!gru/cell/while/gru_cell/BiasAdd_1
gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2!
gru/cell/while/gru_cell/Const_1¡
)gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2+
)gru/cell/while/gru_cell/split_1/split_dim³
gru/cell/while/gru_cell/split_1SplitV*gru/cell/while/gru_cell/BiasAdd_1:output:0(gru/cell/while/gru_cell/Const_1:output:02gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2!
gru/cell/while/gru_cell/split_1¿
gru/cell/while/gru_cell/addAddV2&gru/cell/while/gru_cell/split:output:0(gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add
gru/cell/while/gru_cell/SigmoidSigmoidgru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2!
gru/cell/while/gru_cell/SigmoidÃ
gru/cell/while/gru_cell/add_1AddV2&gru/cell/while/gru_cell/split:output:1(gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_1
!gru/cell/while/gru_cell/Sigmoid_1Sigmoid!gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2#
!gru/cell/while/gru_cell/Sigmoid_1¼
gru/cell/while/gru_cell/mulMul%gru/cell/while/gru_cell/Sigmoid_1:y:0(gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mulº
gru/cell/while/gru_cell/add_2AddV2&gru/cell/while/gru_cell/split:output:2gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_2
gru/cell/while/gru_cell/TanhTanh!gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/Tanh²
gru/cell/while/gru_cell/mul_1Mul#gru/cell/while/gru_cell/Sigmoid:y:0gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_1
gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/while/gru_cell/sub/x¸
gru/cell/while/gru_cell/subSub&gru/cell/while/gru_cell/sub/x:output:0#gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/sub²
gru/cell/while/gru_cell/mul_2Mulgru/cell/while/gru_cell/sub:z:0 gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/mul_2·
gru/cell/while/gru_cell/add_3AddV2!gru/cell/while/gru_cell/mul_1:z:0!gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/while/gru_cell/add_3
3gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemgru_cell_while_placeholder_1gru_cell_while_placeholder!gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype025
3gru/cell/while/TensorArrayV2Write/TensorListSetItemn
gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add/y
gru/cell/while/addAddV2gru_cell_while_placeholdergru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/addr
gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/while/add_1/y£
gru/cell/while/add_1AddV2*gru_cell_while_gru_cell_while_loop_countergru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
gru/cell/while/add_1y
gru/cell/while/IdentityIdentitygru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity
gru/cell/while/Identity_1Identity0gru_cell_while_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
gru/cell/while/Identity_1{
gru/cell/while/Identity_2Identitygru/cell/while/add:z:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_2¨
gru/cell/while/Identity_3IdentityCgru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
gru/cell/while/Identity_3
gru/cell/while/Identity_4Identity!gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
gru/cell/while/Identity_4"ª
Rgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelTgru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Fgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelHgru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
=gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias?gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"P
%gru_cell_while_gru_cell_strided_slice'gru_cell_while_gru_cell_strided_slice_0";
gru_cell_while_identity gru/cell/while/Identity:output:0"?
gru_cell_while_identity_1"gru/cell/while/Identity_1:output:0"?
gru_cell_while_identity_2"gru/cell/while/Identity_2:output:0"?
gru_cell_while_identity_3"gru/cell/while/Identity_3:output:0"?
gru_cell_while_identity_4"gru/cell/while/Identity_4:output:0"Ì
cgru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensoregru_cell_while_tensorarrayv2read_tensorlistgetitem_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
Á
¬
F__inference_functional_1_layer_call_and_return_conditional_losses_3307

inputs
gru_gru_cell_gru_cell_bias 
gru_gru_cell_gru_cell_kernel
gru_gru_cell_variable*
&gru_gru_cell_gru_cell_recurrent_kernel
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢gru/StatefulPartitionedCallÞ
gru/StatefulPartitionedCallStatefulPartitionedCallinputsgru_gru_cell_gru_cell_biasgru_gru_cell_gru_cell_kernelgru_gru_cell_variable&gru_gru_cell_gru_cell_recurrent_kernel*
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
GPU2*0,1J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_31272
gru/StatefulPartitionedCallç
stream/PartitionedCallPartitionedCall$gru/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8 *I
fDRB
@__inference_stream_layer_call_and_return_conditional_losses_31582
stream/PartitionedCallý
dropout/StatefulPartitionedCallStatefulPartitionedCallstream/PartitionedCall:output:0*
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
A__inference_dropout_layer_call_and_return_conditional_losses_31782!
dropout/StatefulPartitionedCall¬
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
?__inference_dense_layer_call_and_return_conditional_losses_32062
dense/StatefulPartitionedCall¸
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_dense_1_kerneldense_1_dense_1_bias*
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
A__inference_dense_1_layer_call_and_return_conditional_losses_32292!
dense_1/StatefulPartitionedCall¹
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_dense_2_kerneldense_2_dense_2_bias*
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
A__inference_dense_2_layer_call_and_return_conditional_losses_32512!
dense_2/StatefulPartitionedCall
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^gru/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2:
gru/StatefulPartitionedCallgru/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs
ÉE

cell_while_body_3035&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2#
cell_while_cell_strided_slice_0a
]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0?
;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0H
Dcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0T
Pcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
cell_while_identity
cell_while_identity_1
cell_while_identity_2
cell_while_identity_3
cell_while_identity_4!
cell_while_cell_strided_slice_
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor=
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasF
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelR
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÍ
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2>
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeè
.cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0cell_while_placeholderEcell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype020
.cell/while/TensorArrayV2Read/TensorListGetItemÅ
"cell/while/gru_cell/ReadVariableOpReadVariableOp;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02$
"cell/while/gru_cell/ReadVariableOp¨
cell/while/gru_cell/unstackUnpack*cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
cell/while/gru_cell/unstackÜ
)cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02+
)cell/while/gru_cell/MatMul/ReadVariableOpÖ
cell/while/gru_cell/MatMulMatMul5cell/while/TensorArrayV2Read/TensorListGetItem:item:01cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/MatMul»
cell/while/gru_cell/BiasAddBiasAdd$cell/while/gru_cell/MatMul:product:0$cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/BiasAddx
cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/gru_cell/Const
#cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#cell/while/gru_cell/split/split_dimä
cell/while/gru_cell/splitSplit,cell/while/gru_cell/split/split_dim:output:0$cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/splití
+cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype02-
+cell/while/gru_cell/MatMul_1/ReadVariableOp¿
cell/while/gru_cell/MatMul_1MatMulcell_while_placeholder_23cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/MatMul_1Á
cell/while/gru_cell/BiasAdd_1BiasAdd&cell/while/gru_cell/MatMul_1:product:0$cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/BiasAdd_1
cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
cell/while/gru_cell/Const_1
%cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%cell/while/gru_cell/split_1/split_dim
cell/while/gru_cell/split_1SplitV&cell/while/gru_cell/BiasAdd_1:output:0$cell/while/gru_cell/Const_1:output:0.cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/split_1¯
cell/while/gru_cell/addAddV2"cell/while/gru_cell/split:output:0$cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add
cell/while/gru_cell/SigmoidSigmoidcell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid³
cell/while/gru_cell/add_1AddV2"cell/while/gru_cell/split:output:1$cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_1
cell/while/gru_cell/Sigmoid_1Sigmoidcell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid_1¬
cell/while/gru_cell/mulMul!cell/while/gru_cell/Sigmoid_1:y:0$cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mulª
cell/while/gru_cell/add_2AddV2"cell/while/gru_cell/split:output:2cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_2
cell/while/gru_cell/TanhTanhcell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Tanh¢
cell/while/gru_cell/mul_1Mulcell/while/gru_cell/Sigmoid:y:0cell_while_placeholder_2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_1{
cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/while/gru_cell/sub/x¨
cell/while/gru_cell/subSub"cell/while/gru_cell/sub/x:output:0cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/sub¢
cell/while/gru_cell/mul_2Mulcell/while/gru_cell/sub:z:0cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_2§
cell/while/gru_cell/add_3AddV2cell/while/gru_cell/mul_1:z:0cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_3õ
/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcell_while_placeholder_1cell_while_placeholdercell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype021
/cell/while/TensorArrayV2Write/TensorListSetItemf
cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add/y}
cell/while/addAddV2cell_while_placeholdercell/while/add/y:output:0*
T0*
_output_shapes
: 2
cell/while/addj
cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add_1/y
cell/while/add_1AddV2"cell_while_cell_while_loop_countercell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cell/while/add_1m
cell/while/IdentityIdentitycell/while/add_1:z:0*
T0*
_output_shapes
: 2
cell/while/Identity
cell/while/Identity_1Identity(cell_while_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
cell/while/Identity_1o
cell/while/Identity_2Identitycell/while/add:z:0*
T0*
_output_shapes
: 2
cell/while/Identity_2
cell/while/Identity_3Identity?cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
cell/while/Identity_3
cell/while/Identity_4Identitycell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
cell/while/Identity_4"@
cell_while_cell_strided_slicecell_while_cell_strided_slice_0"¢
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"x
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"3
cell_while_identitycell/while/Identity:output:0"7
cell_while_identity_1cell/while/Identity_1:output:0"7
cell_while_identity_2cell/while/Identity_2:output:0"7
cell_while_identity_3cell/while/Identity_3:output:0"7
cell_while_identity_4cell/while/Identity_4:output:0"¼
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
³
³
A__inference_dense_2_layer_call_and_return_conditional_losses_3251

inputs(
$matmul_readvariableop_dense_2_kernel'
#biasadd_readvariableop_dense_2_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_2_bias*
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
¸	
É
gru_cell_while_cond_3612.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_3612___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_3612___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_3612___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_3612___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ü

B__inference_gru_cell_layer_call_and_return_conditional_losses_5004

inputs

states)
%readvariableop_gru_cell_gru_cell_bias2
.matmul_readvariableop_gru_cell_gru_cell_kernel@
<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity

identity_1
ReadVariableOpReadVariableOp%readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
ReadVariableOpl
unstackUnpackReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2	
unstack
MatMul/ReadVariableOpReadVariableOp.matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
MatMulk
BiasAddBiasAddMatMul:product:0unstack:output:0*
T0*
_output_shapes
:	°	2	
BiasAddP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constm
split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split/split_dim
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
splits
MatMul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
MatMul_1/ReadVariableOpµ
MatMul_1/ReadVariableOp_1ReadVariableOp<matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02
MatMul_1/ReadVariableOp_1
MatMul_1BatchMatMulV2MatMul_1/ReadVariableOp:value:0!MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:2

MatMul_1i
	BiasAdd_1BiasAddMatMul_1:output:0unstack:output:1*
T0*
_output_shapes
:2
	BiasAdd_1g
Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2	
Const_1q
split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
split_1/split_dim¦
split_1SplitVBiasAdd_1:output:0Const_1:output:0split_1/split_dim:output:0*
T0*

Tlen0* 
_output_shapes
:::*
	num_split2	
split_1X
addAddV2split:output:0split_1:output:0*
T0*
_output_shapes
:2
addI
SigmoidSigmoidadd:z:0*
T0*
_output_shapes
:2	
Sigmoid\
add_1AddV2split:output:1split_1:output:1*
T0*
_output_shapes
:2
add_1O
	Sigmoid_1Sigmoid	add_1:z:0*
T0*
_output_shapes
:2
	Sigmoid_1U
mulMulSigmoid_1:y:0split_1:output:2*
T0*
_output_shapes
:2
mulS
add_2AddV2split:output:2mul:z:0*
T0*
_output_shapes
:2
add_2B
TanhTanh	add_2:z:0*
T0*
_output_shapes
:2
Tanhm
mul_1/ReadVariableOpReadVariableOpstates*
_output_shapes
:*
dtype02
mul_1/ReadVariableOpc
mul_1MulSigmoid:y:0mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:2
mul_1S
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
sub/xQ
subSubsub/x:output:0Sigmoid:y:0*
T0*
_output_shapes
:2
subK
mul_2Mulsub:z:0Tanh:y:0*
T0*
_output_shapes
:2
mul_2P
add_3AddV2	mul_1:z:0	mul_2:z:0*
T0*
_output_shapes
:2
add_3N
IdentityIdentity	add_3:z:0*
T0*
_output_shapes
:2

IdentityR

Identity_1Identity	add_3:z:0*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
::::::F B

_output_shapes

:
 
_user_specified_nameinputs:&"
 
_user_specified_namestates
»>

while_body_4614
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0C
?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0O
Kwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_gru_cell_readvariableop_gru_cell_gru_cell_biasA
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelM
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÊ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¶
while/gru_cell/ReadVariableOpReadVariableOp6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
while/gru_cell/unstackÍ
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÂ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
while/gru_cell/MatMul§
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/splitÞ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp«
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
while/gru_cell/MatMul_1­
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/split_1
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
while/gru_cell/add}
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid_1
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_2v
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Tanh
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
while/gru_cell/sub
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
while/Identity_4"
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"n
4while_gru_cell_readvariableop_gru_cell_gru_cell_bias6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
²

while_cond_4763
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_4763___redundant_placeholder02
.while_while_cond_4763___redundant_placeholder12
.while_while_cond_4763___redundant_placeholder22
.while_while_cond_4763___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ïP
î
>__inference_cell_layer_call_and_return_conditional_losses_4854
inputs_02
.gru_cell_readvariableop_gru_cell_gru_cell_bias;
7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel6
2gru_cell_matmul_1_readvariableop_gru_cell_variableI
Egru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel
identity¢AssignVariableOp¢whileu
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm|
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
	transposeK
ShapeShapetranspose:y:0*
T0*
_output_shapes
:2
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
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape°
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2ó
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
strided_slice_1¢
gru_cell/ReadVariableOpReadVariableOp.gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02
gru_cell/ReadVariableOp
gru_cell/unstackUnpackgru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
gru_cell/unstack¹
gru_cell/MatMul/ReadVariableOpReadVariableOp7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02 
gru_cell/MatMul/ReadVariableOp
gru_cell/MatMulMatMulstrided_slice_1:output:0&gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
gru_cell/MatMul
gru_cell/BiasAddBiasAddgru_cell/MatMul:product:0gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
gru_cell/BiasAddb
gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru_cell/Const
gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split/split_dim¸
gru_cell/splitSplit!gru_cell/split/split_dim:output:0gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/split¸
 gru_cell/MatMul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02"
 gru_cell/MatMul_1/ReadVariableOpÐ
"gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02$
"gru_cell/MatMul_1/ReadVariableOp_1°
gru_cell/MatMul_1MatMul(gru_cell/MatMul_1/ReadVariableOp:value:0*gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
gru_cell/MatMul_1
gru_cell/BiasAdd_1BiasAddgru_cell/MatMul_1:product:0gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
gru_cell/BiasAdd_1y
gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
gru_cell/Const_1
gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
gru_cell/split_1/split_dimè
gru_cell/split_1SplitVgru_cell/BiasAdd_1:output:0gru_cell/Const_1:output:0#gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru_cell/split_1
gru_cell/addAddV2gru_cell/split:output:0gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru_cell/addk
gru_cell/SigmoidSigmoidgru_cell/add:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid
gru_cell/add_1AddV2gru_cell/split:output:1gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru_cell/add_1q
gru_cell/Sigmoid_1Sigmoidgru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru_cell/Sigmoid_1
gru_cell/mulMulgru_cell/Sigmoid_1:y:0gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru_cell/mul~
gru_cell/add_2AddV2gru_cell/split:output:2gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru_cell/add_2d
gru_cell/TanhTanhgru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru_cell/Tanh²
gru_cell/mul_1/ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru_cell/mul_1/ReadVariableOp
gru_cell/mul_1Mulgru_cell/Sigmoid:y:0%gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru_cell/mul_1e
gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru_cell/sub/x|
gru_cell/subSubgru_cell/sub/x:output:0gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru_cell/subv
gru_cell/mul_2Mulgru_cell/sub:z:0gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru_cell/mul_2{
gru_cell/add_3AddV2gru_cell/mul_1:z:0gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru_cell/add_3
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2
TensorArrayV2_1/element_shape¶
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
ReadVariableOpReadVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
ReadVariableOp
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÍ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0ReadVariableOp:value:0strided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0.gru_cell_readvariableop_gru_cell_gru_cell_bias7gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelEgru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*
bodyR
while_body_4764*
condR
while_cond_4763*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     22
0TensorArrayV2Stack/TensorListStack/element_shapeé
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
strided_slice_2y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¦
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*,
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimeó
AssignVariableOpAssignVariableOp2gru_cell_matmul_1_readvariableop_gru_cell_variablewhile:output:4^ReadVariableOp!^gru_cell/MatMul_1/ReadVariableOp^gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOp
IdentityIdentitystrided_slice_2:output:0^AssignVariableOp^while*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::2$
AssignVariableOpAssignVariableOp2
whilewhile:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
»>

while_body_4764
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0:
6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0C
?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0O
Kwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor8
4while_gru_cell_readvariableop_gru_cell_gru_cell_biasA
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelM
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÊ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItem¶
while/gru_cell/ReadVariableOpReadVariableOp6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02
while/gru_cell/ReadVariableOp
while/gru_cell/unstackUnpack%while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
while/gru_cell/unstackÍ
$while/gru_cell/MatMul/ReadVariableOpReadVariableOp?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02&
$while/gru_cell/MatMul/ReadVariableOpÂ
while/gru_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0,while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
while/gru_cell/MatMul§
while/gru_cell/BiasAddBiasAddwhile/gru_cell/MatMul:product:0while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
while/gru_cell/BiasAddn
while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/gru_cell/Const
while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2 
while/gru_cell/split/split_dimÐ
while/gru_cell/splitSplit'while/gru_cell/split/split_dim:output:0while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/splitÞ
&while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype02(
&while/gru_cell/MatMul_1/ReadVariableOp«
while/gru_cell/MatMul_1MatMulwhile_placeholder_2.while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
while/gru_cell/MatMul_1­
while/gru_cell/BiasAdd_1BiasAdd!while/gru_cell/MatMul_1:product:0while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
while/gru_cell/BiasAdd_1
while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
while/gru_cell/Const_1
 while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 while/gru_cell/split_1/split_dim
while/gru_cell/split_1SplitV!while/gru_cell/BiasAdd_1:output:0while/gru_cell/Const_1:output:0)while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
while/gru_cell/split_1
while/gru_cell/addAddV2while/gru_cell/split:output:0while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
while/gru_cell/add}
while/gru_cell/SigmoidSigmoidwhile/gru_cell/add:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid
while/gru_cell/add_1AddV2while/gru_cell/split:output:1while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
while/gru_cell/add_1
while/gru_cell/Sigmoid_1Sigmoidwhile/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Sigmoid_1
while/gru_cell/mulMulwhile/gru_cell/Sigmoid_1:y:0while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
while/gru_cell/mul
while/gru_cell/add_2AddV2while/gru_cell/split:output:2while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_2v
while/gru_cell/TanhTanhwhile/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/Tanh
while/gru_cell/mul_1Mulwhile/gru_cell/Sigmoid:y:0while_placeholder_2*
T0*
_output_shapes
:	2
while/gru_cell/mul_1q
while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
while/gru_cell/sub/x
while/gru_cell/subSubwhile/gru_cell/sub/x:output:0while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
while/gru_cell/sub
while/gru_cell/mul_2Mulwhile/gru_cell/sub:z:0while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
while/gru_cell/mul_2
while/gru_cell/add_3AddV2while/gru_cell/mul_1:z:0while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
while/gru_cell/add_3Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3t
while/Identity_4Identitywhile/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
while/Identity_4"
Iwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelKwhile_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
=while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"n
4while_gru_cell_readvariableop_gru_cell_gru_cell_bias6while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
¸	
É
gru_cell_while_cond_3816.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_3816___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_3816___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_3816___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_3816___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ä
ý
cell_while_cond_3034&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2&
"cell_while_less_cell_strided_slice<
8cell_while_cell_while_cond_3034___redundant_placeholder0<
8cell_while_cell_while_cond_3034___redundant_placeholder1<
8cell_while_cell_while_cond_3034___redundant_placeholder2<
8cell_while_cell_while_cond_3034___redundant_placeholder3
cell_while_identity

cell/while/LessLesscell_while_placeholder"cell_while_less_cell_strided_slice*
T0*
_output_shapes
: 2
cell/while/Lessl
cell/while/IdentityIdentitycell/while/Less:z:0*
T0
*
_output_shapes
: 2
cell/while/Identity"3
cell_while_identitycell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
á"
ß
__inference__traced_save_5221
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop7
3savev2_gru_cell_gru_cell_kernel_read_readvariableopA
=savev2_gru_cell_gru_cell_recurrent_kernel_read_readvariableop5
1savev2_gru_cell_gru_cell_bias_read_readvariableop0
,savev2_gru_cell_variable_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
value3B1 B+_temp_406d281628e34eab93fcd5c3366fe1f3/part2	
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÉ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û
valueÑBÎB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/gru/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop3savev2_gru_cell_gru_cell_kernel_read_readvariableop=savev2_gru_cell_gru_cell_recurrent_kernel_read_readvariableop1savev2_gru_cell_gru_cell_bias_read_readvariableop,savev2_gru_cell_variable_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*{
_input_shapesj
h: :
::
::	::	°	:
°	:	°	:	: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::%!

_output_shapes
:	°	:&"
 
_output_shapes
:
°	:%	!

_output_shapes
:	°	:%
!

_output_shapes
:	:

_output_shapes
: 
Á
Á
__inference__wrapped_model_2228
input_1H
Dfunctional_1_gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasQ
Mfunctional_1_gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelL
Hfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable_
[functional_1_gru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel9
5functional_1_dense_matmul_readvariableop_dense_kernel8
4functional_1_dense_biasadd_readvariableop_dense_bias=
9functional_1_dense_1_matmul_readvariableop_dense_1_kernel<
8functional_1_dense_1_biasadd_readvariableop_dense_1_bias=
9functional_1_dense_2_matmul_readvariableop_dense_2_kernel<
8functional_1_dense_2_biasadd_readvariableop_dense_2_bias
identity¢&functional_1/gru/cell/AssignVariableOp¢functional_1/gru/cell/while¡
$functional_1/gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2&
$functional_1/gru/cell/transpose/perm´
functional_1/gru/cell/transpose	Transposeinput_1-functional_1/gru/cell/transpose/perm:output:0*
T0*"
_output_shapes
:12!
functional_1/gru/cell/transpose
functional_1/gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2
functional_1/gru/cell/Shape 
)functional_1/gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)functional_1/gru/cell/strided_slice/stack¤
+functional_1/gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/gru/cell/strided_slice/stack_1¤
+functional_1/gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+functional_1/gru/cell/strided_slice/stack_2æ
#functional_1/gru/cell/strided_sliceStridedSlice$functional_1/gru/cell/Shape:output:02functional_1/gru/cell/strided_slice/stack:output:04functional_1/gru/cell/strided_slice/stack_1:output:04functional_1/gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#functional_1/gru/cell/strided_slice±
1functional_1/gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ23
1functional_1/gru/cell/TensorArrayV2/element_shape
#functional_1/gru/cell/TensorArrayV2TensorListReserve:functional_1/gru/cell/TensorArrayV2/element_shape:output:0,functional_1/gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02%
#functional_1/gru/cell/TensorArrayV2ë
Kfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2M
Kfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeÐ
=functional_1/gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#functional_1/gru/cell/transpose:y:0Tfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02?
=functional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor¤
+functional_1/gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+functional_1/gru/cell/strided_slice_1/stack¨
-functional_1/gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/gru/cell/strided_slice_1/stack_1¨
-functional_1/gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/gru/cell/strided_slice_1/stack_2÷
%functional_1/gru/cell/strided_slice_1StridedSlice#functional_1/gru/cell/transpose:y:04functional_1/gru/cell/strided_slice_1/stack:output:06functional_1/gru/cell/strided_slice_1/stack_1:output:06functional_1/gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2'
%functional_1/gru/cell/strided_slice_1ä
-functional_1/gru/cell/gru_cell/ReadVariableOpReadVariableOpDfunctional_1_gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02/
-functional_1/gru/cell/gru_cell/ReadVariableOpÉ
&functional_1/gru/cell/gru_cell/unstackUnpack5functional_1/gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2(
&functional_1/gru/cell/gru_cell/unstackû
4functional_1/gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOpMfunctional_1_gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype026
4functional_1/gru/cell/gru_cell/MatMul/ReadVariableOpð
%functional_1/gru/cell/gru_cell/MatMulMatMul.functional_1/gru/cell/strided_slice_1:output:0<functional_1/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2'
%functional_1/gru/cell/gru_cell/MatMulç
&functional_1/gru/cell/gru_cell/BiasAddBiasAdd/functional_1/gru/cell/gru_cell/MatMul:product:0/functional_1/gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2(
&functional_1/gru/cell/gru_cell/BiasAdd
$functional_1/gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/gru/cell/gru_cell/Const«
.functional_1/gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.functional_1/gru/cell/gru_cell/split/split_dim
$functional_1/gru/cell/gru_cell/splitSplit7functional_1/gru/cell/gru_cell/split/split_dim:output:0/functional_1/gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2&
$functional_1/gru/cell/gru_cell/splitú
6functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype028
6functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp
8functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOp[functional_1_gru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02:
8functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp_1
'functional_1/gru/cell/gru_cell/MatMul_1MatMul>functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:0@functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2)
'functional_1/gru/cell/gru_cell/MatMul_1í
(functional_1/gru/cell/gru_cell/BiasAdd_1BiasAdd1functional_1/gru/cell/gru_cell/MatMul_1:product:0/functional_1/gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2*
(functional_1/gru/cell/gru_cell/BiasAdd_1¥
&functional_1/gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2(
&functional_1/gru/cell/gru_cell/Const_1¯
0functional_1/gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ22
0functional_1/gru/cell/gru_cell/split_1/split_dimÖ
&functional_1/gru/cell/gru_cell/split_1SplitV1functional_1/gru/cell/gru_cell/BiasAdd_1:output:0/functional_1/gru/cell/gru_cell/Const_1:output:09functional_1/gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2(
&functional_1/gru/cell/gru_cell/split_1Û
"functional_1/gru/cell/gru_cell/addAddV2-functional_1/gru/cell/gru_cell/split:output:0/functional_1/gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2$
"functional_1/gru/cell/gru_cell/add­
&functional_1/gru/cell/gru_cell/SigmoidSigmoid&functional_1/gru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2(
&functional_1/gru/cell/gru_cell/Sigmoidß
$functional_1/gru/cell/gru_cell/add_1AddV2-functional_1/gru/cell/gru_cell/split:output:1/functional_1/gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/add_1³
(functional_1/gru/cell/gru_cell/Sigmoid_1Sigmoid(functional_1/gru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/gru_cell/Sigmoid_1Ø
"functional_1/gru/cell/gru_cell/mulMul,functional_1/gru/cell/gru_cell/Sigmoid_1:y:0/functional_1/gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2$
"functional_1/gru/cell/gru_cell/mulÖ
$functional_1/gru/cell/gru_cell/add_2AddV2-functional_1/gru/cell/gru_cell/split:output:2&functional_1/gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/add_2¦
#functional_1/gru/cell/gru_cell/TanhTanh(functional_1/gru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2%
#functional_1/gru/cell/gru_cell/Tanhô
3functional_1/gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype025
3functional_1/gru/cell/gru_cell/mul_1/ReadVariableOpæ
$functional_1/gru/cell/gru_cell/mul_1Mul*functional_1/gru/cell/gru_cell/Sigmoid:y:0;functional_1/gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/mul_1
$functional_1/gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2&
$functional_1/gru/cell/gru_cell/sub/xÔ
"functional_1/gru/cell/gru_cell/subSub-functional_1/gru/cell/gru_cell/sub/x:output:0*functional_1/gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2$
"functional_1/gru/cell/gru_cell/subÎ
$functional_1/gru/cell/gru_cell/mul_2Mul&functional_1/gru/cell/gru_cell/sub:z:0'functional_1/gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/mul_2Ó
$functional_1/gru/cell/gru_cell/add_3AddV2(functional_1/gru/cell/gru_cell/mul_1:z:0(functional_1/gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2&
$functional_1/gru/cell/gru_cell/add_3»
3functional_1/gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     25
3functional_1/gru/cell/TensorArrayV2_1/element_shape
%functional_1/gru/cell/TensorArrayV2_1TensorListReserve<functional_1/gru/cell/TensorArrayV2_1/element_shape:output:0,functional_1/gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02'
%functional_1/gru/cell/TensorArrayV2_1z
functional_1/gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
functional_1/gru/cell/timeÖ
$functional_1/gru/cell/ReadVariableOpReadVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02&
$functional_1/gru/cell/ReadVariableOp«
.functional_1/gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ20
.functional_1/gru/cell/while/maximum_iterations
(functional_1/gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2*
(functional_1/gru/cell/while/loop_counter
functional_1/gru/cell/whileWhile1functional_1/gru/cell/while/loop_counter:output:07functional_1/gru/cell/while/maximum_iterations:output:0#functional_1/gru/cell/time:output:0.functional_1/gru/cell/TensorArrayV2_1:handle:0,functional_1/gru/cell/ReadVariableOp:value:0,functional_1/gru/cell/strided_slice:output:0Mfunctional_1/gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:0Dfunctional_1_gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasMfunctional_1_gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel[functional_1_gru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*1
body)R'
%functional_1_gru_cell_while_body_2114*1
cond)R'
%functional_1_gru_cell_while_cond_2113*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
functional_1/gru/cell/whileá
Ffunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2H
Ffunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack/element_shape¸
8functional_1/gru/cell/TensorArrayV2Stack/TensorListStackTensorListStack$functional_1/gru/cell/while:output:3Ofunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02:
8functional_1/gru/cell/TensorArrayV2Stack/TensorListStack­
+functional_1/gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2-
+functional_1/gru/cell/strided_slice_2/stack¨
-functional_1/gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/gru/cell/strided_slice_2/stack_1¨
-functional_1/gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-functional_1/gru/cell/strided_slice_2/stack_2
%functional_1/gru/cell/strided_slice_2StridedSliceAfunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack:tensor:04functional_1/gru/cell/strided_slice_2/stack:output:06functional_1/gru/cell/strided_slice_2/stack_1:output:06functional_1/gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2'
%functional_1/gru/cell/strided_slice_2¥
&functional_1/gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2(
&functional_1/gru/cell/transpose_1/permõ
!functional_1/gru/cell/transpose_1	TransposeAfunctional_1/gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0/functional_1/gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12#
!functional_1/gru/cell/transpose_1
functional_1/gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
functional_1/gru/cell/runtime
&functional_1/gru/cell/AssignVariableOpAssignVariableOpHfunctional_1_gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable$functional_1/gru/cell/while:output:4%^functional_1/gru/cell/ReadVariableOp7^functional_1/gru/cell/gru_cell/MatMul_1/ReadVariableOp4^functional_1/gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02(
&functional_1/gru/cell/AssignVariableOp
functional_1/gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
functional_1/gru/ExpandDims/dimÐ
functional_1/gru/ExpandDims
ExpandDims.functional_1/gru/cell/strided_slice_2:output:0(functional_1/gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
functional_1/gru/ExpandDims
!functional_1/stream/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2#
!functional_1/stream/flatten/ConstÑ
#functional_1/stream/flatten/ReshapeReshape$functional_1/gru/ExpandDims:output:0*functional_1/stream/flatten/Const:output:0*
T0*
_output_shapes
:	2%
#functional_1/stream/flatten/Reshape¢
functional_1/dropout/IdentityIdentity,functional_1/stream/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/IdentityÌ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp5functional_1_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpÄ
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/MatMulÈ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpÅ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/BiasAddÔ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp9functional_1_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpÇ
functional_1/dense_1/MatMulMatMul#functional_1/dense/BiasAdd:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/MatMulÐ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpÍ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/dense_1/ReluÓ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp9functional_1_dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpÊ
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/MatMulÏ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp8functional_1_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpÌ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/BiasAdd·
IdentityIdentity%functional_1/dense_2/BiasAdd:output:0'^functional_1/gru/cell/AssignVariableOp^functional_1/gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::2P
&functional_1/gru/cell/AssignVariableOp&functional_1/gru/cell/AssignVariableOp2:
functional_1/gru/cell/whilefunctional_1/gru/cell/while:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!
_user_specified_name	input_1
¸	
É
gru_cell_while_cond_3997.
*gru_cell_while_gru_cell_while_loop_counter4
0gru_cell_while_gru_cell_while_maximum_iterations
gru_cell_while_placeholder 
gru_cell_while_placeholder_1 
gru_cell_while_placeholder_2.
*gru_cell_while_less_gru_cell_strided_sliceD
@gru_cell_while_gru_cell_while_cond_3997___redundant_placeholder0D
@gru_cell_while_gru_cell_while_cond_3997___redundant_placeholder1D
@gru_cell_while_gru_cell_while_cond_3997___redundant_placeholder2D
@gru_cell_while_gru_cell_while_cond_3997___redundant_placeholder3
gru_cell_while_identity

gru/cell/while/LessLessgru_cell_while_placeholder*gru_cell_while_less_gru_cell_strided_slice*
T0*
_output_shapes
: 2
gru/cell/while/Lessx
gru/cell/while/IdentityIdentitygru/cell/while/Less:z:0*
T0
*
_output_shapes
: 2
gru/cell/while/Identity";
gru_cell_while_identity gru/cell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Á	
à
'__inference_gru_cell_layer_call_fn_5012

inputs
states_0
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity

identity_1¢StatefulPartitionedCall½
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0gru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

::*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_50042
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*-
_input_shapes
:::::22
StatefulPartitionedCallStatefulPartitionedCall:F B

_output_shapes

:
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0
Ð

&__inference_dense_1_layer_call_fn_4537

inputs
dense_1_kernel
dense_1_bias
identity¢StatefulPartitionedCallø
StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_kerneldense_1_bias*
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
A__inference_dense_1_layer_call_and_return_conditional_losses_32292
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
Ä
ý
cell_while_cond_4201&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2&
"cell_while_less_cell_strided_slice<
8cell_while_cell_while_cond_4201___redundant_placeholder0<
8cell_while_cell_while_cond_4201___redundant_placeholder1<
8cell_while_cell_while_cond_4201___redundant_placeholder2<
8cell_while_cell_while_cond_4201___redundant_placeholder3
cell_while_identity

cell/while/LessLesscell_while_placeholder"cell_while_less_cell_strided_slice*
T0*
_output_shapes
: 2
cell/while/Lessl
cell/while/IdentityIdentitycell/while/Less:z:0*
T0
*
_output_shapes
: 2
cell/while/Identity"3
cell_while_identitycell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
ÿ
Ô
"__inference_gru_layer_call_fn_4455

inputs
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel
gru_cell_variable&
"gru_cell_gru_cell_recurrent_kernel
identity¢StatefulPartitionedCallÆ
StatefulPartitionedCallStatefulPartitionedCallinputsgru_cell_gru_cell_biasgru_cell_gru_cell_kernelgru_cell_variable"gru_cell_gru_cell_recurrent_kernel*
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
GPU2*0,1J 8 *F
fAR?
=__inference_gru_layer_call_and_return_conditional_losses_31272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*#
_output_shapes
:2

Identity"
identityIdentity:output:0*1
_input_shapes 
:1::::22
StatefulPartitionedCallStatefulPartitionedCall:J F
"
_output_shapes
:1
 
_user_specified_nameinputs
Ù]
á
%functional_1_gru_cell_while_body_2114H
Dfunctional_1_gru_cell_while_functional_1_gru_cell_while_loop_counterN
Jfunctional_1_gru_cell_while_functional_1_gru_cell_while_maximum_iterations+
'functional_1_gru_cell_while_placeholder-
)functional_1_gru_cell_while_placeholder_1-
)functional_1_gru_cell_while_placeholder_2E
Afunctional_1_gru_cell_while_functional_1_gru_cell_strided_slice_0
functional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensor_0P
Lfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0Y
Ufunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0e
afunctional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0(
$functional_1_gru_cell_while_identity*
&functional_1_gru_cell_while_identity_1*
&functional_1_gru_cell_while_identity_2*
&functional_1_gru_cell_while_identity_3*
&functional_1_gru_cell_while_identity_4C
?functional_1_gru_cell_while_functional_1_gru_cell_strided_slice
}functional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensorN
Jfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasW
Sfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelc
_functional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelï
Mfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2O
Mfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeÎ
?functional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemfunctional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensor_0'functional_1_gru_cell_while_placeholderVfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype02A
?functional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItemø
3functional_1/gru/cell/while/gru_cell/ReadVariableOpReadVariableOpLfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype025
3functional_1/gru/cell/while/gru_cell/ReadVariableOpÛ
,functional_1/gru/cell/while/gru_cell/unstackUnpack;functional_1/gru/cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2.
,functional_1/gru/cell/while/gru_cell/unstack
:functional_1/gru/cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpUfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02<
:functional_1/gru/cell/while/gru_cell/MatMul/ReadVariableOp
+functional_1/gru/cell/while/gru_cell/MatMulMatMulFfunctional_1/gru/cell/while/TensorArrayV2Read/TensorListGetItem:item:0Bfunctional_1/gru/cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2-
+functional_1/gru/cell/while/gru_cell/MatMulÿ
,functional_1/gru/cell/while/gru_cell/BiasAddBiasAdd5functional_1/gru/cell/while/gru_cell/MatMul:product:05functional_1/gru/cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2.
,functional_1/gru/cell/while/gru_cell/BiasAdd
*functional_1/gru/cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2,
*functional_1/gru/cell/while/gru_cell/Const·
4functional_1/gru/cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ26
4functional_1/gru/cell/while/gru_cell/split/split_dim¨
*functional_1/gru/cell/while/gru_cell/splitSplit=functional_1/gru/cell/while/gru_cell/split/split_dim:output:05functional_1/gru/cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2,
*functional_1/gru/cell/while/gru_cell/split 
<functional_1/gru/cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpafunctional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype02>
<functional_1/gru/cell/while/gru_cell/MatMul_1/ReadVariableOp
-functional_1/gru/cell/while/gru_cell/MatMul_1MatMul)functional_1_gru_cell_while_placeholder_2Dfunctional_1/gru/cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2/
-functional_1/gru/cell/while/gru_cell/MatMul_1
.functional_1/gru/cell/while/gru_cell/BiasAdd_1BiasAdd7functional_1/gru/cell/while/gru_cell/MatMul_1:product:05functional_1/gru/cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	20
.functional_1/gru/cell/while/gru_cell/BiasAdd_1±
,functional_1/gru/cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2.
,functional_1/gru/cell/while/gru_cell/Const_1»
6functional_1/gru/cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ28
6functional_1/gru/cell/while/gru_cell/split_1/split_dimô
,functional_1/gru/cell/while/gru_cell/split_1SplitV7functional_1/gru/cell/while/gru_cell/BiasAdd_1:output:05functional_1/gru/cell/while/gru_cell/Const_1:output:0?functional_1/gru/cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2.
,functional_1/gru/cell/while/gru_cell/split_1ó
(functional_1/gru/cell/while/gru_cell/addAddV23functional_1/gru/cell/while/gru_cell/split:output:05functional_1/gru/cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/while/gru_cell/add¿
,functional_1/gru/cell/while/gru_cell/SigmoidSigmoid,functional_1/gru/cell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2.
,functional_1/gru/cell/while/gru_cell/Sigmoid÷
*functional_1/gru/cell/while/gru_cell/add_1AddV23functional_1/gru/cell/while/gru_cell/split:output:15functional_1/gru/cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/add_1Å
.functional_1/gru/cell/while/gru_cell/Sigmoid_1Sigmoid.functional_1/gru/cell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	20
.functional_1/gru/cell/while/gru_cell/Sigmoid_1ð
(functional_1/gru/cell/while/gru_cell/mulMul2functional_1/gru/cell/while/gru_cell/Sigmoid_1:y:05functional_1/gru/cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/while/gru_cell/mulî
*functional_1/gru/cell/while/gru_cell/add_2AddV23functional_1/gru/cell/while/gru_cell/split:output:2,functional_1/gru/cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/add_2¸
)functional_1/gru/cell/while/gru_cell/TanhTanh.functional_1/gru/cell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2+
)functional_1/gru/cell/while/gru_cell/Tanhæ
*functional_1/gru/cell/while/gru_cell/mul_1Mul0functional_1/gru/cell/while/gru_cell/Sigmoid:y:0)functional_1_gru_cell_while_placeholder_2*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/mul_1
*functional_1/gru/cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2,
*functional_1/gru/cell/while/gru_cell/sub/xì
(functional_1/gru/cell/while/gru_cell/subSub3functional_1/gru/cell/while/gru_cell/sub/x:output:00functional_1/gru/cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2*
(functional_1/gru/cell/while/gru_cell/subæ
*functional_1/gru/cell/while/gru_cell/mul_2Mul,functional_1/gru/cell/while/gru_cell/sub:z:0-functional_1/gru/cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/mul_2ë
*functional_1/gru/cell/while/gru_cell/add_3AddV2.functional_1/gru/cell/while/gru_cell/mul_1:z:0.functional_1/gru/cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2,
*functional_1/gru/cell/while/gru_cell/add_3Ê
@functional_1/gru/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)functional_1_gru_cell_while_placeholder_1'functional_1_gru_cell_while_placeholder.functional_1/gru/cell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype02B
@functional_1/gru/cell/while/TensorArrayV2Write/TensorListSetItem
!functional_1/gru/cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2#
!functional_1/gru/cell/while/add/yÁ
functional_1/gru/cell/while/addAddV2'functional_1_gru_cell_while_placeholder*functional_1/gru/cell/while/add/y:output:0*
T0*
_output_shapes
: 2!
functional_1/gru/cell/while/add
#functional_1/gru/cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2%
#functional_1/gru/cell/while/add_1/yä
!functional_1/gru/cell/while/add_1AddV2Dfunctional_1_gru_cell_while_functional_1_gru_cell_while_loop_counter,functional_1/gru/cell/while/add_1/y:output:0*
T0*
_output_shapes
: 2#
!functional_1/gru/cell/while/add_1 
$functional_1/gru/cell/while/IdentityIdentity%functional_1/gru/cell/while/add_1:z:0*
T0*
_output_shapes
: 2&
$functional_1/gru/cell/while/IdentityÉ
&functional_1/gru/cell/while/Identity_1IdentityJfunctional_1_gru_cell_while_functional_1_gru_cell_while_maximum_iterations*
T0*
_output_shapes
: 2(
&functional_1/gru/cell/while/Identity_1¢
&functional_1/gru/cell/while/Identity_2Identity#functional_1/gru/cell/while/add:z:0*
T0*
_output_shapes
: 2(
&functional_1/gru/cell/while/Identity_2Ï
&functional_1/gru/cell/while/Identity_3IdentityPfunctional_1/gru/cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2(
&functional_1/gru/cell/while/Identity_3¶
&functional_1/gru/cell/while/Identity_4Identity.functional_1/gru/cell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2(
&functional_1/gru/cell/while/Identity_4"
?functional_1_gru_cell_while_functional_1_gru_cell_strided_sliceAfunctional_1_gru_cell_while_functional_1_gru_cell_strided_slice_0"Ä
_functional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelafunctional_1_gru_cell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"¬
Sfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelUfunctional_1_gru_cell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"
Jfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasLfunctional_1_gru_cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"U
$functional_1_gru_cell_while_identity-functional_1/gru/cell/while/Identity:output:0"Y
&functional_1_gru_cell_while_identity_1/functional_1/gru/cell/while/Identity_1:output:0"Y
&functional_1_gru_cell_while_identity_2/functional_1/gru/cell/while/Identity_2:output:0"Y
&functional_1_gru_cell_while_identity_3/functional_1/gru/cell/while/Identity_3:output:0"Y
&functional_1_gru_cell_while_identity_4/functional_1/gru/cell/while/Identity_4:output:0"
}functional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensorfunctional_1_gru_cell_while_tensorarrayv2read_tensorlistgetitem_functional_1_gru_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
ÉE

cell_while_body_4202&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2#
cell_while_cell_strided_slice_0a
]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0?
;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0H
Dcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0T
Pcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0
cell_while_identity
cell_while_identity_1
cell_while_identity_2
cell_while_identity_3
cell_while_identity_4!
cell_while_cell_strided_slice_
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor=
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_biasF
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelR
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelÍ
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2>
<cell/while/TensorArrayV2Read/TensorListGetItem/element_shapeè
.cell/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0cell_while_placeholderEcell/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype020
.cell/while/TensorArrayV2Read/TensorListGetItemÅ
"cell/while/gru_cell/ReadVariableOpReadVariableOp;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0*
_output_shapes
:	°	*
dtype02$
"cell/while/gru_cell/ReadVariableOp¨
cell/while/gru_cell/unstackUnpack*cell/while/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
cell/while/gru_cell/unstackÜ
)cell/while/gru_cell/MatMul/ReadVariableOpReadVariableOpDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0*
_output_shapes
:	°	*
dtype02+
)cell/while/gru_cell/MatMul/ReadVariableOpÖ
cell/while/gru_cell/MatMulMatMul5cell/while/TensorArrayV2Read/TensorListGetItem:item:01cell/while/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/MatMul»
cell/while/gru_cell/BiasAddBiasAdd$cell/while/gru_cell/MatMul:product:0$cell/while/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/BiasAddx
cell/while/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/gru_cell/Const
#cell/while/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#cell/while/gru_cell/split/split_dimä
cell/while/gru_cell/splitSplit,cell/while/gru_cell/split/split_dim:output:0$cell/while/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/splití
+cell/while/gru_cell/MatMul_1/ReadVariableOpReadVariableOpPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0* 
_output_shapes
:
°	*
dtype02-
+cell/while/gru_cell/MatMul_1/ReadVariableOp¿
cell/while/gru_cell/MatMul_1MatMulcell_while_placeholder_23cell/while/gru_cell/MatMul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/MatMul_1Á
cell/while/gru_cell/BiasAdd_1BiasAdd&cell/while/gru_cell/MatMul_1:product:0$cell/while/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
cell/while/gru_cell/BiasAdd_1
cell/while/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
cell/while/gru_cell/Const_1
%cell/while/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2'
%cell/while/gru_cell/split_1/split_dim
cell/while/gru_cell/split_1SplitV&cell/while/gru_cell/BiasAdd_1:output:0$cell/while/gru_cell/Const_1:output:0.cell/while/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
cell/while/gru_cell/split_1¯
cell/while/gru_cell/addAddV2"cell/while/gru_cell/split:output:0$cell/while/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add
cell/while/gru_cell/SigmoidSigmoidcell/while/gru_cell/add:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid³
cell/while/gru_cell/add_1AddV2"cell/while/gru_cell/split:output:1$cell/while/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_1
cell/while/gru_cell/Sigmoid_1Sigmoidcell/while/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Sigmoid_1¬
cell/while/gru_cell/mulMul!cell/while/gru_cell/Sigmoid_1:y:0$cell/while/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mulª
cell/while/gru_cell/add_2AddV2"cell/while/gru_cell/split:output:2cell/while/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_2
cell/while/gru_cell/TanhTanhcell/while/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/Tanh¢
cell/while/gru_cell/mul_1Mulcell/while/gru_cell/Sigmoid:y:0cell_while_placeholder_2*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_1{
cell/while/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
cell/while/gru_cell/sub/x¨
cell/while/gru_cell/subSub"cell/while/gru_cell/sub/x:output:0cell/while/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/sub¢
cell/while/gru_cell/mul_2Mulcell/while/gru_cell/sub:z:0cell/while/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/mul_2§
cell/while/gru_cell/add_3AddV2cell/while/gru_cell/mul_1:z:0cell/while/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
cell/while/gru_cell/add_3õ
/cell/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemcell_while_placeholder_1cell_while_placeholdercell/while/gru_cell/add_3:z:0*
_output_shapes
: *
element_dtype021
/cell/while/TensorArrayV2Write/TensorListSetItemf
cell/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add/y}
cell/while/addAddV2cell_while_placeholdercell/while/add/y:output:0*
T0*
_output_shapes
: 2
cell/while/addj
cell/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cell/while/add_1/y
cell/while/add_1AddV2"cell_while_cell_while_loop_countercell/while/add_1/y:output:0*
T0*
_output_shapes
: 2
cell/while/add_1m
cell/while/IdentityIdentitycell/while/add_1:z:0*
T0*
_output_shapes
: 2
cell/while/Identity
cell/while/Identity_1Identity(cell_while_cell_while_maximum_iterations*
T0*
_output_shapes
: 2
cell/while/Identity_1o
cell/while/Identity_2Identitycell/while/add:z:0*
T0*
_output_shapes
: 2
cell/while/Identity_2
cell/while/Identity_3Identity?cell/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
cell/while/Identity_3
cell/while/Identity_4Identitycell/while/gru_cell/add_3:z:0*
T0*
_output_shapes
:	2
cell/while/Identity_4"@
cell_while_cell_strided_slicecell_while_cell_strided_slice_0"¢
Ncell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernelPcell_while_gru_cell_matmul_1_readvariableop_gru_cell_gru_cell_recurrent_kernel_0"
Bcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelDcell_while_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel_0"x
9cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias;cell_while_gru_cell_readvariableop_gru_cell_gru_cell_bias_0"3
cell_while_identitycell/while/Identity:output:0"7
cell_while_identity_1cell/while/Identity_1:output:0"7
cell_while_identity_2cell/while/Identity_2:output:0"7
cell_while_identity_3cell/while/Identity_3:output:0"7
cell_while_identity_4cell/while/Identity_4:output:0"¼
[cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor]cell_while_tensorarrayv2read_tensorlistgetitem_cell_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 

×
#__inference_cell_layer_call_fn_4872
inputs_0
gru_cell_variable
gru_cell_gru_cell_bias
gru_cell_gru_cell_kernel&
"gru_cell_gru_cell_recurrent_kernel
identity¢StatefulPartitionedCallÅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0gru_cell_variablegru_cell_gru_cell_biasgru_cell_gru_cell_kernel"gru_cell_gru_cell_recurrent_kernel*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *G
fBR@
>__inference_cell_layer_call_and_return_conditional_losses_28122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0

³
A__inference_dense_1_layer_call_and_return_conditional_losses_4530

inputs(
$matmul_readvariableop_dense_1_kernel'
#biasadd_readvariableop_dense_1_bias
identity
MatMul/ReadVariableOpReadVariableOp$matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp#biasadd_readvariableop_dense_1_bias*
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
û-
º
 __inference__traced_restore_5261
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias/
+assignvariableop_6_gru_cell_gru_cell_kernel9
5assignvariableop_7_gru_cell_gru_cell_recurrent_kernel-
)assignvariableop_8_gru_cell_gru_cell_bias(
$assignvariableop_9_gru_cell_variable
identity_11¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ï
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Û
valueÑBÎB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEBFlayer_with_weights-0/gru/keras_api/states/0/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¤
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*)
value BB B B B B B B B B B B 2
RestoreV2/shape_and_slicesâ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*@
_output_shapes.
,:::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¦
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOp+assignvariableop_6_gru_cell_gru_cell_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7º
AssignVariableOp_7AssignVariableOp5assignvariableop_7_gru_cell_gru_cell_recurrent_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8®
AssignVariableOp_8AssignVariableOp)assignvariableop_8_gru_cell_gru_cell_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9©
AssignVariableOp_9AssignVariableOp$assignvariableop_9_gru_cell_variableIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº
Identity_10Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_10­
Identity_11IdentityIdentity_10:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_11"#
identity_11Identity_11:output:0*=
_input_shapes,
*: ::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
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
¤
_
A__inference_dropout_layer_call_and_return_conditional_losses_3183

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
#
¼
while_body_2652
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_strided_slice_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0+
'while_gru_cell_gru_cell_gru_cell_bias_0-
)while_gru_cell_gru_cell_gru_cell_kernel_07
3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_strided_sliceU
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor)
%while_gru_cell_gru_cell_gru_cell_bias+
'while_gru_cell_gru_cell_gru_cell_kernel5
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel¢&while/gru_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÊ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
_output_shapes

:*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÐ
&while/gru_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2'while_gru_cell_gru_cell_gru_cell_bias_0)while_gru_cell_gru_cell_gru_cell_kernel_03while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0*
Tin	
2*
Tout
2*
_collective_manager_ids
 **
_output_shapes
:	:	*%
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8 *K
fFRD
B__inference_gru_cell_layer_call_and_return_conditional_losses_24282(
&while/gru_cell/StatefulPartitionedCalló
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder/while/gru_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2¶
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3´
while/Identity_4Identity/while/gru_cell/StatefulPartitionedCall:output:1'^while/gru_cell/StatefulPartitionedCall*
T0*
_output_shapes
:	2
while/Identity_4"P
%while_gru_cell_gru_cell_gru_cell_bias'while_gru_cell_gru_cell_gru_cell_bias_0"T
'while_gru_cell_gru_cell_gru_cell_kernel)while_gru_cell_gru_cell_gru_cell_kernel_0"h
1while_gru_cell_gru_cell_gru_cell_recurrent_kernel3while_gru_cell_gru_cell_gru_cell_recurrent_kernel_0")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0",
while_strided_slicewhile_strided_slice_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*6
_input_shapes%
#: : : : :	: : :::2P
&while/gru_cell/StatefulPartitionedCall&while/gru_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
: 
®
­
?__inference_dense_layer_call_and_return_conditional_losses_4512

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpk
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
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
Ä
ý
cell_while_cond_4353&
"cell_while_cell_while_loop_counter,
(cell_while_cell_while_maximum_iterations
cell_while_placeholder
cell_while_placeholder_1
cell_while_placeholder_2&
"cell_while_less_cell_strided_slice<
8cell_while_cell_while_cond_4353___redundant_placeholder0<
8cell_while_cell_while_cond_4353___redundant_placeholder1<
8cell_while_cell_while_cond_4353___redundant_placeholder2<
8cell_while_cell_while_cond_4353___redundant_placeholder3
cell_while_identity

cell/while/LessLesscell_while_placeholder"cell_while_less_cell_strided_slice*
T0*
_output_shapes
: 2
cell/while/Lessl
cell/while/IdentityIdentitycell/while/Less:z:0*
T0
*
_output_shapes
: 2
cell/while/Identity"3
cell_while_identitycell/while/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ø

`
A__inference_dropout_layer_call_and_return_conditional_losses_4487

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
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
dropout/Shape¬
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
 *ÍÌÌ=2
dropout/GreaterEqual/y¶
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
Àw
Ì
F__inference_functional_1_layer_call_and_return_conditional_losses_3727
input_1;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity¢gru/cell/AssignVariableOp¢gru/cell/while
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposeinput_1 gru/cell/transpose/perm:output:0*
T0*"
_output_shapes
:12
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$gru/cell/TensorArrayV2/element_shapeÔ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ñ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2©
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp¢
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
gru/cell/gru_cell/unstackÔ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOp¼
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/gru_cell/split/split_dimÜ
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitÓ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpë
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Ô
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul_1¹
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1¤
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul¢
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhÍ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3¡
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2(
&gru/cell/TensorArrayV2_1/element_shapeÚ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time¯
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterË
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_3613*$
condR
gru_cell_while_cond_3612*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileÇ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2È
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permÁ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims}
stream/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream/flatten/Const
stream/flatten/ReshapeReshapegru/ExpandDims:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshape{
dropout/IdentityIdentitystream/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity¥
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul¡
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
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
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:T P
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
!
_user_specified_name	input_1
²

while_cond_2651
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_less_strided_slice2
.while_while_cond_2651___redundant_placeholder02
.while_while_cond_2651___redundant_placeholder12
.while_while_cond_2651___redundant_placeholder22
.while_while_cond_2651___redundant_placeholder3
while_identity
n

while/LessLesswhile_placeholderwhile_less_strided_slice*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*8
_input_shapes'
%: : : : :	: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	:

_output_shapes
: :

_output_shapes
:
Ã

$__inference_dense_layer_call_fn_4519

inputs
dense_kernel

dense_bias
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsdense_kernel
dense_bias*
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
?__inference_dense_layer_call_and_return_conditional_losses_32062
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
¤
Ë
F__inference_functional_1_layer_call_and_return_conditional_losses_3938

inputs;
7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_biasD
@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel?
;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variableR
Ngru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity¢gru/cell/AssignVariableOp¢gru/cell/while
gru/cell/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose/perm
gru/cell/transpose	Transposeinputs gru/cell/transpose/perm:output:0*
T0*"
_output_shapes
:12
gru/cell/transposeu
gru/cell/ShapeConst*
_output_shapes
:*
dtype0*!
valueB"1         2
gru/cell/Shape
gru/cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
gru/cell/strided_slice/stack
gru/cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_1
gru/cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
gru/cell/strided_slice/stack_2
gru/cell/strided_sliceStridedSlicegru/cell/Shape:output:0%gru/cell/strided_slice/stack:output:0'gru/cell/strided_slice/stack_1:output:0'gru/cell/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gru/cell/strided_slice
$gru/cell/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2&
$gru/cell/TensorArrayV2/element_shapeÔ
gru/cell/TensorArrayV2TensorListReserve-gru/cell/TensorArrayV2/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2Ñ
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"      2@
>gru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape
0gru/cell/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorgru/cell/transpose:y:0Ggru/cell/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type022
0gru/cell/TensorArrayUnstack/TensorListFromTensor
gru/cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
gru/cell/strided_slice_1/stack
 gru/cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_1
 gru/cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_1/stack_2©
gru/cell/strided_slice_1StridedSlicegru/cell/transpose:y:0'gru/cell/strided_slice_1/stack:output:0)gru/cell/strided_slice_1/stack_1:output:0)gru/cell/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
shrink_axis_mask2
gru/cell/strided_slice_1½
 gru/cell/gru_cell/ReadVariableOpReadVariableOp7gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias*
_output_shapes
:	°	*
dtype02"
 gru/cell/gru_cell/ReadVariableOp¢
gru/cell/gru_cell/unstackUnpack(gru/cell/gru_cell/ReadVariableOp:value:0*
T0*"
_output_shapes
:°	:°	*	
num2
gru/cell/gru_cell/unstackÔ
'gru/cell/gru_cell/MatMul/ReadVariableOpReadVariableOp@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernel*
_output_shapes
:	°	*
dtype02)
'gru/cell/gru_cell/MatMul/ReadVariableOp¼
gru/cell/gru_cell/MatMulMatMul!gru/cell/strided_slice_1:output:0/gru/cell/gru_cell/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul³
gru/cell/gru_cell/BiasAddBiasAdd"gru/cell/gru_cell/MatMul:product:0"gru/cell/gru_cell/unstack:output:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAddt
gru/cell/gru_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
gru/cell/gru_cell/Const
!gru/cell/gru_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/gru_cell/split/split_dimÜ
gru/cell/gru_cell/splitSplit*gru/cell/gru_cell/split/split_dim:output:0"gru/cell/gru_cell/BiasAdd:output:0*
T0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/splitÓ
)gru/cell/gru_cell/MatMul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02+
)gru/cell/gru_cell/MatMul_1/ReadVariableOpë
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1ReadVariableOpNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel* 
_output_shapes
:
°	*
dtype02-
+gru/cell/gru_cell/MatMul_1/ReadVariableOp_1Ô
gru/cell/gru_cell/MatMul_1MatMul1gru/cell/gru_cell/MatMul_1/ReadVariableOp:value:03gru/cell/gru_cell/MatMul_1/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/MatMul_1¹
gru/cell/gru_cell/BiasAdd_1BiasAdd$gru/cell/gru_cell/MatMul_1:product:0"gru/cell/gru_cell/unstack:output:1*
T0*
_output_shapes
:	°	2
gru/cell/gru_cell/BiasAdd_1
gru/cell/gru_cell/Const_1Const*
_output_shapes
:*
dtype0*!
valueB"    ÿÿÿÿ2
gru/cell/gru_cell/Const_1
#gru/cell/gru_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2%
#gru/cell/gru_cell/split_1/split_dim
gru/cell/gru_cell/split_1SplitV$gru/cell/gru_cell/BiasAdd_1:output:0"gru/cell/gru_cell/Const_1:output:0,gru/cell/gru_cell/split_1/split_dim:output:0*
T0*

Tlen0*5
_output_shapes#
!:	:	:	*
	num_split2
gru/cell/gru_cell/split_1§
gru/cell/gru_cell/addAddV2 gru/cell/gru_cell/split:output:0"gru/cell/gru_cell/split_1:output:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add
gru/cell/gru_cell/SigmoidSigmoidgru/cell/gru_cell/add:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid«
gru/cell/gru_cell/add_1AddV2 gru/cell/gru_cell/split:output:1"gru/cell/gru_cell/split_1:output:1*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_1
gru/cell/gru_cell/Sigmoid_1Sigmoidgru/cell/gru_cell/add_1:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/Sigmoid_1¤
gru/cell/gru_cell/mulMulgru/cell/gru_cell/Sigmoid_1:y:0"gru/cell/gru_cell/split_1:output:2*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul¢
gru/cell/gru_cell/add_2AddV2 gru/cell/gru_cell/split:output:2gru/cell/gru_cell/mul:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_2
gru/cell/gru_cell/TanhTanhgru/cell/gru_cell/add_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/TanhÍ
&gru/cell/gru_cell/mul_1/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02(
&gru/cell/gru_cell/mul_1/ReadVariableOp²
gru/cell/gru_cell/mul_1Mulgru/cell/gru_cell/Sigmoid:y:0.gru/cell/gru_cell/mul_1/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_1w
gru/cell/gru_cell/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
gru/cell/gru_cell/sub/x 
gru/cell/gru_cell/subSub gru/cell/gru_cell/sub/x:output:0gru/cell/gru_cell/Sigmoid:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/sub
gru/cell/gru_cell/mul_2Mulgru/cell/gru_cell/sub:z:0gru/cell/gru_cell/Tanh:y:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/mul_2
gru/cell/gru_cell/add_3AddV2gru/cell/gru_cell/mul_1:z:0gru/cell/gru_cell/mul_2:z:0*
T0*
_output_shapes
:	2
gru/cell/gru_cell/add_3¡
&gru/cell/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2(
&gru/cell/TensorArrayV2_1/element_shapeÚ
gru/cell/TensorArrayV2_1TensorListReserve/gru/cell/TensorArrayV2_1/element_shape:output:0gru/cell/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
gru/cell/TensorArrayV2_1`
gru/cell/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/time¯
gru/cell/ReadVariableOpReadVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variable*
_output_shapes
:	*
dtype02
gru/cell/ReadVariableOp
!gru/cell/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2#
!gru/cell/while/maximum_iterations|
gru/cell/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
gru/cell/while/loop_counterË
gru/cell/whileWhile$gru/cell/while/loop_counter:output:0*gru/cell/while/maximum_iterations:output:0gru/cell/time:output:0!gru/cell/TensorArrayV2_1:handle:0gru/cell/ReadVariableOp:value:0gru/cell/strided_slice:output:0@gru/cell/TensorArrayUnstack/TensorListFromTensor:output_handle:07gru_cell_gru_cell_readvariableop_gru_cell_gru_cell_bias@gru_cell_gru_cell_matmul_readvariableop_gru_cell_gru_cell_kernelNgru_cell_gru_cell_matmul_1_readvariableop_1_gru_cell_gru_cell_recurrent_kernel*
T
2
*
_lower_using_switch_merge(*
_num_original_outputs
*1
_output_shapes
: : : : :	: : : : : *%
_read_only_resource_inputs
	*$
bodyR
gru_cell_while_body_3817*$
condR
gru_cell_while_cond_3816*0
output_shapes
: : : : :	: : : : : *
parallel_iterations 2
gru/cell/whileÇ
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"     2;
9gru/cell/TensorArrayV2Stack/TensorListStack/element_shape
+gru/cell/TensorArrayV2Stack/TensorListStackTensorListStackgru/cell/while:output:3Bgru/cell/TensorArrayV2Stack/TensorListStack/element_shape:output:0*#
_output_shapes
:1*
element_dtype02-
+gru/cell/TensorArrayV2Stack/TensorListStack
gru/cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2 
gru/cell/strided_slice_2/stack
 gru/cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2"
 gru/cell/strided_slice_2/stack_1
 gru/cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 gru/cell/strided_slice_2/stack_2È
gru/cell/strided_slice_2StridedSlice4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0'gru/cell/strided_slice_2/stack:output:0)gru/cell/strided_slice_2/stack_1:output:0)gru/cell/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:	*
shrink_axis_mask2
gru/cell/strided_slice_2
gru/cell/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
gru/cell/transpose_1/permÁ
gru/cell/transpose_1	Transpose4gru/cell/TensorArrayV2Stack/TensorListStack:tensor:0"gru/cell/transpose_1/perm:output:0*
T0*#
_output_shapes
:12
gru/cell/transpose_1x
gru/cell/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
gru/cell/runtime²
gru/cell/AssignVariableOpAssignVariableOp;gru_cell_gru_cell_matmul_1_readvariableop_gru_cell_variablegru/cell/while:output:4^gru/cell/ReadVariableOp*^gru/cell/gru_cell/MatMul_1/ReadVariableOp'^gru/cell/gru_cell/mul_1/ReadVariableOp*
_output_shapes
 *
dtype02
gru/cell/AssignVariableOpj
gru/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
gru/ExpandDims/dim
gru/ExpandDims
ExpandDims!gru/cell/strided_slice_2:output:0gru/ExpandDims/dim:output:0*
T0*#
_output_shapes
:2
gru/ExpandDims}
stream/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  2
stream/flatten/Const
stream/flatten/ReshapeReshapegru/ExpandDims:output:0stream/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
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
dropout/dropout/ShapeÄ
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
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yÖ
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
dropout/dropout/Mul_1¥
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul¡
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/BiasAdd­
dense_1/MatMul/ReadVariableOpReadVariableOp,dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/BiasAdd:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense_1/MatMul©
dense_1/BiasAdd/ReadVariableOpReadVariableOp+dense_1_biasadd_readvariableop_dense_1_bias*
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
dense_1/Relu¬
dense_2/MatMul/ReadVariableOpReadVariableOp,dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02
dense_2/MatMul/ReadVariableOp
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/MatMul¨
dense_2/BiasAdd/ReadVariableOpReadVariableOp+dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense_2/BiasAdd
IdentityIdentitydense_2/BiasAdd:output:0^gru/cell/AssignVariableOp^gru/cell/while*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:1::::::::::26
gru/cell/AssignVariableOpgru/cell/AssignVariableOp2 
gru/cell/whilegru/cell/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ1
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
6
input_1+
serving_default_input_1:012
dense_2'
StatefulPartitionedCall:0tensorflow/serving/predict:
1
layer-0
layer_with_weights-0
layer-1
layer-2
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
s_default_save_signature
t__call__
*u&call_and_return_all_conditional_losses".
_tf_keras_networkð-{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "mode": "TRAINING", "inference_batch_size": 1, "units": 400, "return_sequences": 0, "unroll": false, "stateful": true}, "name": "gru", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 400], "ring_buffer_size_in_time_dim": null}, "name": "stream", "inbound_nodes": [[["gru", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 49, 13]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "GRU", "config": {"name": "gru", "trainable": true, "dtype": "float32", "mode": "TRAINING", "inference_batch_size": 1, "units": 400, "return_sequences": 0, "unroll": false, "stateful": true}, "name": "gru", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 400], "ring_buffer_size_in_time_dim": null}, "name": "stream", "inbound_nodes": [[["gru", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
í"ê
_tf_keras_input_layerÊ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 13]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 13]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
¥
gru
trainable_variables
regularization_losses
	variables
	keras_api
v__call__
*w&call_and_return_all_conditional_losses"
_tf_keras_layeró{"class_name": "GRU", "name": "gru", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "gru", "trainable": true, "dtype": "float32", "mode": "TRAINING", "inference_batch_size": 1, "units": 400, "return_sequences": 0, "unroll": false, "stateful": true}}
È
cell
state_shape
trainable_variables
regularization_losses
	variables
	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer{"class_name": "Stream", "name": "stream", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 400], "ring_buffer_size_in_time_dim": null}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 400]}}
á
trainable_variables
regularization_losses
	variables
	keras_api
z__call__
*{&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
§

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layerè{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 400}}}}
©

"kernel
#bias
$trainable_variables
%regularization_losses
&	variables
'	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layerê{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¬

(kernel
)bias
*trainable_variables
+regularization_losses
,	variables
-	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerë{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
_
.0
/1
02
3
4
"5
#6
(7
)8"
trackable_list_wrapper
 "
trackable_list_wrapper
_
.0
/1
02
3
4
"5
#6
(7
)8"
trackable_list_wrapper
Ê
1layer_metrics

2layers
trainable_variables
3layer_regularization_losses
4metrics
5non_trainable_variables
	regularization_losses

	variables
t__call__
s_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
à

6cell
7
state_spec
8trainable_variables
9regularization_losses
:	variables
;	keras_api
__call__
+&call_and_return_all_conditional_losses"µ	
_tf_keras_rnn_layer	{"class_name": "GRU", "name": "cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": true, "must_restore_from_config": false, "config": {"name": "cell", "trainable": true, "dtype": "float32", "return_sequences": 0, "return_state": false, "go_backwards": false, "stateful": true, "unroll": false, "time_major": false, "units": 400, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [1, null, 13]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}]}
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
­
<layer_metrics

=layers
trainable_variables
>layer_regularization_losses
?metrics
@non_trainable_variables
regularization_losses
	variables
v__call__
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
ä
Atrainable_variables
Bregularization_losses
C	variables
D	keras_api
__call__
+&call_and_return_all_conditional_losses"Ó
_tf_keras_layer¹{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Elayer_metrics

Flayers
trainable_variables
Glayer_regularization_losses
Hmetrics
Inon_trainable_variables
regularization_losses
	variables
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jlayer_metrics

Klayers
trainable_variables
Llayer_regularization_losses
Mmetrics
Nnon_trainable_variables
regularization_losses
	variables
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Olayer_metrics

Players
trainable_variables
Qlayer_regularization_losses
Rmetrics
Snon_trainable_variables
regularization_losses
 	variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
­
Tlayer_metrics

Ulayers
$trainable_variables
Vlayer_regularization_losses
Wmetrics
Xnon_trainable_variables
%regularization_losses
&	variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
°
Ylayer_metrics

Zlayers
*trainable_variables
[layer_regularization_losses
\metrics
]non_trainable_variables
+regularization_losses
,	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
+:)	°	2gru/cell/gru_cell/kernel
6:4
°	2"gru/cell/gru_cell/recurrent_kernel
):'	°	2gru/cell/gru_cell/bias
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
 "
trackable_list_wrapper


.kernel
/recurrent_kernel
0bias
^trainable_variables
_regularization_losses
`	variables
a	keras_api
__call__
+&call_and_return_all_conditional_losses"â
_tf_keras_layerÈ{"class_name": "GRUCell", "name": "gru_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gru_cell", "trainable": true, "dtype": "float32", "units": 400, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2, "reset_after": true}}
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
¼
blayer_metrics

clayers
8trainable_variables
dlayer_regularization_losses
emetrics
fnon_trainable_variables
9regularization_losses

gstates
:	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
hlayer_metrics

ilayers
Atrainable_variables
jlayer_regularization_losses
kmetrics
lnon_trainable_variables
Bregularization_losses
C	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
0"
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
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
°
mlayer_metrics

nlayers
^trainable_variables
olayer_regularization_losses
pmetrics
qnon_trainable_variables
_regularization_losses
`	variables
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
r0"
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
$:"	2gru/cell/Variable
á2Þ
__inference__wrapped_model_2228º
²
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
annotationsª **¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ1
ú2÷
+__inference_functional_1_layer_call_fn_4142
+__inference_functional_1_layer_call_fn_3742
+__inference_functional_1_layer_call_fn_4127
+__inference_functional_1_layer_call_fn_3757À
·²³
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
kwonlydefaultsª 
annotationsª *
 
æ2ã
F__inference_functional_1_layer_call_and_return_conditional_losses_3727
F__inference_functional_1_layer_call_and_return_conditional_losses_4112
F__inference_functional_1_layer_call_and_return_conditional_losses_3938
F__inference_functional_1_layer_call_and_return_conditional_losses_3553À
·²³
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
kwonlydefaultsª 
annotationsª *
 
2þ
"__inference_gru_layer_call_fn_4464
"__inference_gru_layer_call_fn_4455³
ª²¦
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
annotationsª *
 
·2´
=__inference_gru_layer_call_and_return_conditional_losses_4446
=__inference_gru_layer_call_and_return_conditional_losses_4294³
ª²¦
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
annotationsª *
 
Ï2Ì
%__inference_stream_layer_call_fn_4475¢
²
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
annotationsª *
 
ê2ç
@__inference_stream_layer_call_and_return_conditional_losses_4470¢
²
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
annotationsª *
 
2
&__inference_dropout_layer_call_fn_4497
&__inference_dropout_layer_call_fn_4502´
«²§
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
kwonlydefaultsª 
annotationsª *
 
À2½
A__inference_dropout_layer_call_and_return_conditional_losses_4492
A__inference_dropout_layer_call_and_return_conditional_losses_4487´
«²§
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
kwonlydefaultsª 
annotationsª *
 
Î2Ë
$__inference_dense_layer_call_fn_4519¢
²
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
annotationsª *
 
é2æ
?__inference_dense_layer_call_and_return_conditional_losses_4512¢
²
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
annotationsª *
 
Ð2Í
&__inference_dense_1_layer_call_fn_4537¢
²
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
annotationsª *
 
ë2è
A__inference_dense_1_layer_call_and_return_conditional_losses_4530¢
²
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
annotationsª *
 
Ð2Í
&__inference_dense_2_layer_call_fn_4554¢
²
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
annotationsª *
 
ë2è
A__inference_dense_2_layer_call_and_return_conditional_losses_4547¢
²
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
annotationsª *
 
1B/
"__inference_signature_wrapper_3372input_1
¥2¢
#__inference_cell_layer_call_fn_4872
#__inference_cell_layer_call_fn_4863Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Û2Ø
>__inference_cell_layer_call_and_return_conditional_losses_4704
>__inference_cell_layer_call_and_return_conditional_losses_4854Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
¨2¥¢
²
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
annotationsª *
 
è2å
'__inference_gru_cell_layer_call_fn_5066
'__inference_gru_cell_layer_call_fn_5012
'__inference_gru_cell_layer_call_fn_5168
'__inference_gru_cell_layer_call_fn_5157¾
µ²±
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
kwonlydefaultsª 
annotationsª *
 
Ô2Ñ
B__inference_gru_cell_layer_call_and_return_conditional_losses_4915
B__inference_gru_cell_layer_call_and_return_conditional_losses_4958
B__inference_gru_cell_layer_call_and_return_conditional_losses_5106
B__inference_gru_cell_layer_call_and_return_conditional_losses_5146¾
µ²±
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
kwonlydefaultsª 
annotationsª *
 
__inference__wrapped_model_2228l
0.r/"#()4¢1
*¢'
%"
input_1ÿÿÿÿÿÿÿÿÿ1
ª "(ª%
#
dense_2
dense_2¯
>__inference_cell_layer_call_and_return_conditional_losses_4704m0.r/F¢C
<¢9
+(
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

 
p

 
ª "¢

0	
 ¯
>__inference_cell_layer_call_and_return_conditional_losses_4854m0.r/F¢C
<¢9
+(
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "¢

0	
 
#__inference_cell_layer_call_fn_4863`r0./F¢C
<¢9
+(
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

 
p

 
ª "	
#__inference_cell_layer_call_fn_4872`r0./F¢C
<¢9
+(
&#
inputs/0ÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "	
A__inference_dense_1_layer_call_and_return_conditional_losses_4530L"#'¢$
¢

inputs	
ª "¢

0	
 i
&__inference_dense_1_layer_call_fn_4537?"#'¢$
¢

inputs	
ª "	
A__inference_dense_2_layer_call_and_return_conditional_losses_4547K()'¢$
¢

inputs	
ª "¢

0
 h
&__inference_dense_2_layer_call_fn_4554>()'¢$
¢

inputs	
ª "
?__inference_dense_layer_call_and_return_conditional_losses_4512L'¢$
¢

inputs	
ª "¢

0	
 g
$__inference_dense_layer_call_fn_4519?'¢$
¢

inputs	
ª "	
A__inference_dropout_layer_call_and_return_conditional_losses_4487L+¢(
!¢

inputs	
p
ª "¢

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_4492L+¢(
!¢

inputs	
p 
ª "¢

0	
 i
&__inference_dropout_layer_call_fn_4497?+¢(
!¢

inputs	
p
ª "	i
&__inference_dropout_layer_call_fn_4502?+¢(
!¢

inputs	
p 
ª "	²
F__inference_functional_1_layer_call_and_return_conditional_losses_3553h
0.r/"#()<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ1
p

 
ª "¢

0
 ²
F__inference_functional_1_layer_call_and_return_conditional_losses_3727h
0.r/"#()<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ1
p 

 
ª "¢

0
 ±
F__inference_functional_1_layer_call_and_return_conditional_losses_3938g
0.r/"#();¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ1
p

 
ª "¢

0
 ±
F__inference_functional_1_layer_call_and_return_conditional_losses_4112g
0.r/"#();¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 

 
ª "¢

0
 
+__inference_functional_1_layer_call_fn_3742[
0.r/"#()<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ1
p

 
ª "
+__inference_functional_1_layer_call_fn_3757[
0.r/"#()<¢9
2¢/
%"
input_1ÿÿÿÿÿÿÿÿÿ1
p 

 
ª "
+__inference_functional_1_layer_call_fn_4127Z
0.r/"#();¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ1
p

 
ª "
+__inference_functional_1_layer_call_fn_4142Z
0.r/"#();¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ1
p 

 
ª "ì
B__inference_gru_cell_layer_call_and_return_conditional_losses_4915¥0./h¢e
^¢[

inputs
<¢9
74	"¢
ú	


jstates/0VariableSpec
p
ª "4¢1
*¢'

0/0


0/1/0
 ì
B__inference_gru_cell_layer_call_and_return_conditional_losses_4958¥0./h¢e
^¢[

inputs
<¢9
74	"¢
ú	


jstates/0VariableSpec
p 
ª "4¢1
*¢'

0/0


0/1/0
 Ý
B__inference_gru_cell_layer_call_and_return_conditional_losses_51060./K¢H
A¢>

inputs
¢

states/0	
p
ª "B¢?
8¢5

0/0	


0/1/0	
 Ý
B__inference_gru_cell_layer_call_and_return_conditional_losses_51460./K¢H
A¢>

inputs
¢

states/0	
p 
ª "B¢?
8¢5

0/0	


0/1/0	
 Ã
'__inference_gru_cell_layer_call_fn_50120./h¢e
^¢[

inputs
<¢9
74	"¢
ú	


jstates/0VariableSpec
p
ª "&¢#
	
0


1/0Ã
'__inference_gru_cell_layer_call_fn_50660./h¢e
^¢[

inputs
<¢9
74	"¢
ú	


jstates/0VariableSpec
p 
ª "&¢#
	
0


1/0´
'__inference_gru_cell_layer_call_fn_51570./K¢H
A¢>

inputs
¢

states/0	
p
ª "4¢1

0	


1/0	´
'__inference_gru_cell_layer_call_fn_51680./K¢H
A¢>

inputs
¢

states/0	
p 
ª "4¢1

0	


1/0	
=__inference_gru_layer_call_and_return_conditional_losses_4294Y0.r/.¢+
$¢!

inputs1
p
ª "!¢

0
 
=__inference_gru_layer_call_and_return_conditional_losses_4446Y0.r/.¢+
$¢!

inputs1
p 
ª "!¢

0
 r
"__inference_gru_layer_call_fn_4455L0.r/.¢+
$¢!

inputs1
p
ª "r
"__inference_gru_layer_call_fn_4464L0.r/.¢+
$¢!

inputs1
p 
ª "
"__inference_signature_wrapper_3372n
0.r/"#()6¢3
¢ 
,ª)
'
input_1
input_11"(ª%
#
dense_2
dense_2
@__inference_stream_layer_call_and_return_conditional_losses_4470L+¢(
!¢

inputs
ª "¢

0	
 h
%__inference_stream_layer_call_fn_4475?+¢(
!¢

inputs
ª "	