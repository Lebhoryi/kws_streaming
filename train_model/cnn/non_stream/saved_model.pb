Εδ
Ρ£
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
Ύ
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878α

v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
*
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

stream/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*%
shared_namestream/conv2d/kernel

(stream/conv2d/kernel/Read/ReadVariableOpReadVariableOpstream/conv2d/kernel*&
_output_shapes
:@*
dtype0
|
stream/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*#
shared_namestream/conv2d/bias
u
&stream/conv2d/bias/Read/ReadVariableOpReadVariableOpstream/conv2d/bias*
_output_shapes
:@*
dtype0

stream_1/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_namestream_1/conv2d_1/kernel

,stream_1/conv2d_1/kernel/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/kernel*&
_output_shapes
:@@*
dtype0

stream_1/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_1/conv2d_1/bias
}
*stream_1/conv2d_1/bias/Read/ReadVariableOpReadVariableOpstream_1/conv2d_1/bias*
_output_shapes
:@*
dtype0

stream_2/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_namestream_2/conv2d_2/kernel

,stream_2/conv2d_2/kernel/Read/ReadVariableOpReadVariableOpstream_2/conv2d_2/kernel*&
_output_shapes
:@@*
dtype0

stream_2/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_2/conv2d_2/bias
}
*stream_2/conv2d_2/bias/Read/ReadVariableOpReadVariableOpstream_2/conv2d_2/bias*
_output_shapes
:@*
dtype0

stream_3/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_namestream_3/conv2d_3/kernel

,stream_3/conv2d_3/kernel/Read/ReadVariableOpReadVariableOpstream_3/conv2d_3/kernel*&
_output_shapes
:@@*
dtype0

stream_3/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_3/conv2d_3/bias
}
*stream_3/conv2d_3/bias/Read/ReadVariableOpReadVariableOpstream_3/conv2d_3/bias*
_output_shapes
:@*
dtype0

stream_4/conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namestream_4/conv2d_4/kernel

,stream_4/conv2d_4/kernel/Read/ReadVariableOpReadVariableOpstream_4/conv2d_4/kernel*'
_output_shapes
:@*
dtype0

stream_4/conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestream_4/conv2d_4/bias
~
*stream_4/conv2d_4/bias/Read/ReadVariableOpReadVariableOpstream_4/conv2d_4/bias*
_output_shapes	
:*
dtype0

stream_5/conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namestream_5/conv2d_5/kernel

,stream_5/conv2d_5/kernel/Read/ReadVariableOpReadVariableOpstream_5/conv2d_5/kernel*'
_output_shapes
:@*
dtype0

stream_5/conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_namestream_5/conv2d_5/bias
}
*stream_5/conv2d_5/bias/Read/ReadVariableOpReadVariableOpstream_5/conv2d_5/bias*
_output_shapes
:@*
dtype0

stream_6/conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*)
shared_namestream_6/conv2d_6/kernel

,stream_6/conv2d_6/kernel/Read/ReadVariableOpReadVariableOpstream_6/conv2d_6/kernel*'
_output_shapes
:@*
dtype0

stream_6/conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestream_6/conv2d_6/bias
~
*stream_6/conv2d_6/bias/Read/ReadVariableOpReadVariableOpstream_6/conv2d_6/bias*
_output_shapes	
:*
dtype0

NoOpNoOp
M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ΣL
valueΙLBΖL BΏL
 
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api

signatures
 
R
	variables
trainable_variables
regularization_losses
	keras_api
m
cell
state_shape
	variables
trainable_variables
regularization_losses
	keras_api
m
cell
state_shape
 	variables
!trainable_variables
"regularization_losses
#	keras_api
m
$cell
%state_shape
&	variables
'trainable_variables
(regularization_losses
)	keras_api
m
*cell
+state_shape
,	variables
-trainable_variables
.regularization_losses
/	keras_api
m
0cell
1state_shape
2	variables
3trainable_variables
4regularization_losses
5	keras_api
m
6cell
7state_shape
8	variables
9trainable_variables
:regularization_losses
;	keras_api
m
<cell
=state_shape
>	variables
?trainable_variables
@regularization_losses
A	keras_api
m
Bcell
Cstate_shape
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
R
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
h

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
h

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api

^0
_1
`2
a3
b4
c5
d6
e7
f8
g9
h10
i11
j12
k13
L14
M15
R16
S17
X18
Y19

^0
_1
`2
a3
b4
c5
d6
e7
f8
g9
h10
i11
j12
k13
L14
M15
R16
S17
X18
Y19
 
­
	variables
lnon_trainable_variables
mmetrics
nlayer_regularization_losses

olayers
player_metrics
trainable_variables
regularization_losses
 
 
 
 
­
qnon_trainable_variables
	variables
rlayer_metrics

slayers
trainable_variables
tlayer_regularization_losses
umetrics
regularization_losses
h

^kernel
_bias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
 

^0
_1

^0
_1
 
­
znon_trainable_variables
	variables
{layer_metrics

|layers
trainable_variables
}layer_regularization_losses
~metrics
regularization_losses
k

`kernel
abias
	variables
trainable_variables
regularization_losses
	keras_api
 

`0
a1

`0
a1
 
²
non_trainable_variables
 	variables
layer_metrics
layers
!trainable_variables
 layer_regularization_losses
metrics
"regularization_losses
l

bkernel
cbias
	variables
trainable_variables
regularization_losses
	keras_api
 

b0
c1

b0
c1
 
²
non_trainable_variables
&	variables
layer_metrics
layers
'trainable_variables
 layer_regularization_losses
metrics
(regularization_losses
l

dkernel
ebias
	variables
trainable_variables
regularization_losses
	keras_api
 

d0
e1

d0
e1
 
²
non_trainable_variables
,	variables
layer_metrics
layers
-trainable_variables
 layer_regularization_losses
metrics
.regularization_losses
l

fkernel
gbias
	variables
trainable_variables
regularization_losses
	keras_api
 

f0
g1

f0
g1
 
²
non_trainable_variables
2	variables
layer_metrics
 layers
3trainable_variables
 ‘layer_regularization_losses
’metrics
4regularization_losses
l

hkernel
ibias
£	variables
€trainable_variables
₯regularization_losses
¦	keras_api
 

h0
i1

h0
i1
 
²
§non_trainable_variables
8	variables
¨layer_metrics
©layers
9trainable_variables
 ͺlayer_regularization_losses
«metrics
:regularization_losses
l

jkernel
kbias
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
 

j0
k1

j0
k1
 
²
°non_trainable_variables
>	variables
±layer_metrics
²layers
?trainable_variables
 ³layer_regularization_losses
΄metrics
@regularization_losses
V
΅	variables
Άtrainable_variables
·regularization_losses
Έ	keras_api
 
 
 
 
²
Ήnon_trainable_variables
D	variables
Ίlayer_metrics
»layers
Etrainable_variables
 Όlayer_regularization_losses
½metrics
Fregularization_losses
 
 
 
²
Ύnon_trainable_variables
H	variables
Ώlayer_metrics
ΐlayers
Itrainable_variables
 Αlayer_regularization_losses
Βmetrics
Jregularization_losses
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
²
Γnon_trainable_variables
N	variables
Δlayer_metrics
Εlayers
Otrainable_variables
 Ζlayer_regularization_losses
Ηmetrics
Pregularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
 
²
Θnon_trainable_variables
T	variables
Ιlayer_metrics
Κlayers
Utrainable_variables
 Λlayer_regularization_losses
Μmetrics
Vregularization_losses
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

X0
Y1
 
²
Νnon_trainable_variables
Z	variables
Ξlayer_metrics
Οlayers
[trainable_variables
 Πlayer_regularization_losses
Ρmetrics
\regularization_losses
PN
VARIABLE_VALUEstream/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
NL
VARIABLE_VALUEstream/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_1/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_1/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_2/conv2d_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_2/conv2d_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_3/conv2d_3/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_3/conv2d_3/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEstream_4/conv2d_4/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEstream_4/conv2d_4/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEstream_5/conv2d_5/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_5/conv2d_5/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEstream_6/conv2d_6/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEstream_6/conv2d_6/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
f
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
11
12
13
 
 
 
 
 
 

^0
_1

^0
_1
 
²
?non_trainable_variables
v	variables
Σlayer_metrics
Τlayers
wtrainable_variables
 Υlayer_regularization_losses
Φmetrics
xregularization_losses
 
 

0
 
 

`0
a1

`0
a1
 
΄
Χnon_trainable_variables
	variables
Ψlayer_metrics
Ωlayers
trainable_variables
 Ϊlayer_regularization_losses
Ϋmetrics
regularization_losses
 
 

0
 
 

b0
c1

b0
c1
 
΅
άnon_trainable_variables
	variables
έlayer_metrics
ήlayers
trainable_variables
 ίlayer_regularization_losses
ΰmetrics
regularization_losses
 
 

$0
 
 

d0
e1

d0
e1
 
΅
αnon_trainable_variables
	variables
βlayer_metrics
γlayers
trainable_variables
 δlayer_regularization_losses
εmetrics
regularization_losses
 
 

*0
 
 

f0
g1

f0
g1
 
΅
ζnon_trainable_variables
	variables
ηlayer_metrics
θlayers
trainable_variables
 ιlayer_regularization_losses
κmetrics
regularization_losses
 
 

00
 
 

h0
i1

h0
i1
 
΅
λnon_trainable_variables
£	variables
μlayer_metrics
νlayers
€trainable_variables
 ξlayer_regularization_losses
οmetrics
₯regularization_losses
 
 

60
 
 

j0
k1

j0
k1
 
΅
πnon_trainable_variables
¬	variables
ρlayer_metrics
ςlayers
­trainable_variables
 σlayer_regularization_losses
τmetrics
?regularization_losses
 
 

<0
 
 
 
 
 
΅
υnon_trainable_variables
΅	variables
φlayer_metrics
χlayers
Άtrainable_variables
 ψlayer_regularization_losses
ωmetrics
·regularization_losses
 
 

B0
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
p
serving_default_input_1Placeholder*"
_output_shapes
:*
dtype0*
shape:
ϊ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1stream/conv2d/kernelstream/conv2d/biasstream_1/conv2d_1/kernelstream_1/conv2d_1/biasstream_2/conv2d_2/kernelstream_2/conv2d_2/biasstream_3/conv2d_3/kernelstream_3/conv2d_3/biasstream_4/conv2d_4/kernelstream_4/conv2d_4/biasstream_5/conv2d_5/kernelstream_5/conv2d_5/biasstream_6/conv2d_6/kernelstream_6/conv2d_6/biasdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_signature_wrapper_817
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ι
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp(stream/conv2d/kernel/Read/ReadVariableOp&stream/conv2d/bias/Read/ReadVariableOp,stream_1/conv2d_1/kernel/Read/ReadVariableOp*stream_1/conv2d_1/bias/Read/ReadVariableOp,stream_2/conv2d_2/kernel/Read/ReadVariableOp*stream_2/conv2d_2/bias/Read/ReadVariableOp,stream_3/conv2d_3/kernel/Read/ReadVariableOp*stream_3/conv2d_3/bias/Read/ReadVariableOp,stream_4/conv2d_4/kernel/Read/ReadVariableOp*stream_4/conv2d_4/bias/Read/ReadVariableOp,stream_5/conv2d_5/kernel/Read/ReadVariableOp*stream_5/conv2d_5/bias/Read/ReadVariableOp,stream_6/conv2d_6/kernel/Read/ReadVariableOp*stream_6/conv2d_6/bias/Read/ReadVariableOpConst*!
Tin
2*
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
__inference__traced_save_1549
Τ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasstream/conv2d/kernelstream/conv2d/biasstream_1/conv2d_1/kernelstream_1/conv2d_1/biasstream_2/conv2d_2/kernelstream_2/conv2d_2/biasstream_3/conv2d_3/kernelstream_3/conv2d_3/biasstream_4/conv2d_4/kernelstream_4/conv2d_4/biasstream_5/conv2d_5/kernelstream_5/conv2d_5/biasstream_6/conv2d_6/kernelstream_6/conv2d_6/bias* 
Tin
2*
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
 __inference__traced_restore_1619ͺ	
Ύ

Ω
A__inference_stream_6_layer_call_and_return_conditional_losses_503

inputs;
7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel:
6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias
identityΑ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp·
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_6/Conv2DΆ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp€
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_6/BiasAdds
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_6/Reluo
IdentityIdentityconv2d_6/Relu:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs


Μ
@__inference_stream_layer_call_and_return_conditional_losses_1261

inputs5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identityΆ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp°
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d/Conv2D«
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d/Relul
IdentityIdentityconv2d/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
::::N J
&
_output_shapes
:
 
_user_specified_nameinputs
»

Ω
A__inference_stream_5_layer_call_and_return_conditional_losses_480

inputs;
7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel:
6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias
identityΑ
conv2d_5/Conv2D/ReadVariableOpReadVariableOp7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpΆ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_5/Conv2D΅
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp£
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_5/BiasAddr
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_5/Relun
IdentityIdentityconv2d_5/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*.
_input_shapes
::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ή

Ϊ
B__inference_stream_3_layer_call_and_return_conditional_losses_1315

inputs;
7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel:
6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias
identityΐ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΆ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_3/Conv2D΅
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp£
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_3/BiasAddr
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_3/Relun
IdentityIdentityconv2d_3/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
­
¬
>__inference_dense_layer_call_and_return_conditional_losses_570

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
Ό

Ϊ
B__inference_stream_5_layer_call_and_return_conditional_losses_1351

inputs;
7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel:
6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias
identityΑ
conv2d_5/Conv2D/ReadVariableOpReadVariableOp7conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02 
conv2d_5/Conv2D/ReadVariableOpΆ
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_5/Conv2D΅
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp6conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp£
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_5/BiasAddr
conv2d_5/ReluReluconv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_5/Relun
IdentityIdentityconv2d_5/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*.
_input_shapes
::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Θ
^
B__inference_stream_7_layer_call_and_return_conditional_losses_1382

inputs
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Constw
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshaped
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs


'__inference_stream_4_layer_call_fn_1340

inputs
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_4_conv2d_4_kernelstream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_4_layer_call_and_return_conditional_losses_4572
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*-
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ήZ
Γ

E__inference_functional_1_layer_call_and_return_conditional_losses_978
input_1<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_biasD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_biasD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_biasD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_biasD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΝ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/ReluΫ
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpο
stream_2/conv2d_2/Conv2DConv2D$stream_1/conv2d_1/Relu:activations:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2DΠ
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpΗ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/ReluΫ
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpο
stream_3/conv2d_3/Conv2DConv2D$stream_2/conv2d_2/Relu:activations:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2DΠ
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpΗ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Reluά
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpπ
stream_4/conv2d_4/Conv2DConv2D$stream_3/conv2d_3/Relu:activations:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2DΡ
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpΘ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Reluά
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpο
stream_5/conv2d_5/Conv2DConv2D$stream_4/conv2d_4/Relu:activations:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2DΠ
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpΗ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Reluά
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpπ
stream_6/conv2d_6/Conv2DConv2D$stream_5/conv2d_5/Relu:activations:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2DΡ
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpΘ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu
stream_7/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
stream_7/flatten/Const°
stream_7/flatten/ReshapeReshape$stream_6/conv2d_6/Relu:activations:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshape}
dropout/IdentityIdentity!stream_7/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
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
dense_2/BiasAddc
IdentityIdentitydense_2/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^::::::::::::::::::::::T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
Ώ

Ϊ
B__inference_stream_6_layer_call_and_return_conditional_losses_1369

inputs;
7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel:
6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias
identityΑ
conv2d_6/Conv2D/ReadVariableOpReadVariableOp7conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02 
conv2d_6/Conv2D/ReadVariableOp·
conv2d_6/Conv2DConv2Dinputs&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_6/Conv2DΆ
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp6conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp€
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_6/BiasAdds
conv2d_6/ReluReluconv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_6/Reluo
IdentityIdentityconv2d_6/Relu:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
2
	
__inference__traced_save_1549
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop3
/savev2_stream_conv2d_kernel_read_readvariableop1
-savev2_stream_conv2d_bias_read_readvariableop7
3savev2_stream_1_conv2d_1_kernel_read_readvariableop5
1savev2_stream_1_conv2d_1_bias_read_readvariableop7
3savev2_stream_2_conv2d_2_kernel_read_readvariableop5
1savev2_stream_2_conv2d_2_bias_read_readvariableop7
3savev2_stream_3_conv2d_3_kernel_read_readvariableop5
1savev2_stream_3_conv2d_3_bias_read_readvariableop7
3savev2_stream_4_conv2d_4_kernel_read_readvariableop5
1savev2_stream_4_conv2d_4_bias_read_readvariableop7
3savev2_stream_5_conv2d_5_kernel_read_readvariableop5
1savev2_stream_5_conv2d_5_bias_read_readvariableop7
3savev2_stream_6_conv2d_6_kernel_read_readvariableop5
1savev2_stream_6_conv2d_6_bias_read_readvariableop
savev2_const

identity_1’MergeV2Checkpoints
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
value3B1 B+_temp_6da9aaa222a44fd58dc11296891fb111/part2	
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*±
value§B€B6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names²
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop/savev2_stream_conv2d_kernel_read_readvariableop-savev2_stream_conv2d_bias_read_readvariableop3savev2_stream_1_conv2d_1_kernel_read_readvariableop1savev2_stream_1_conv2d_1_bias_read_readvariableop3savev2_stream_2_conv2d_2_kernel_read_readvariableop1savev2_stream_2_conv2d_2_bias_read_readvariableop3savev2_stream_3_conv2d_3_kernel_read_readvariableop1savev2_stream_3_conv2d_3_bias_read_readvariableop3savev2_stream_4_conv2d_4_kernel_read_readvariableop1savev2_stream_4_conv2d_4_bias_read_readvariableop3savev2_stream_5_conv2d_5_kernel_read_readvariableop1savev2_stream_5_conv2d_5_bias_read_readvariableop3savev2_stream_6_conv2d_6_kernel_read_readvariableop1savev2_stream_6_conv2d_6_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2Ί
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes‘
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

identity_1Identity_1:output:0*ύ
_input_shapesλ
θ: :
::
::	::@:@:@@:@:@@:@:@@:@:@::@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!
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
::,(
&
_output_shapes
:@: 

_output_shapes
:@:,	(
&
_output_shapes
:@@: 


_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::-)
'
_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:!

_output_shapes	
::

_output_shapes
: 
Κ

&__inference_dense_1_layer_call_fn_1449

inputs
dense_1_kernel
dense_1_bias
identity’StatefulPartitionedCallς
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_5932
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


'__inference_stream_6_layer_call_fn_1376

inputs
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_6_conv2d_6_kernelstream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_6_layer_call_and_return_conditional_losses_5032
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*-
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ώ

Ϊ
B__inference_stream_4_layer_call_and_return_conditional_losses_1333

inputs;
7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel:
6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias
identityΑ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp·
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_4/Conv2DΆ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp€
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_4/BiasAdds
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_4/Reluo
IdentityIdentityconv2d_4/Relu:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs

³
A__inference_dense_1_layer_call_and_return_conditional_losses_1442

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
’A
?
E__inference_functional_1_layer_call_and_return_conditional_losses_705

inputs
stream_stream_conv2d_kernel
stream_stream_conv2d_bias%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias%
!stream_2_stream_2_conv2d_2_kernel#
stream_2_stream_2_conv2d_2_bias%
!stream_3_stream_3_conv2d_3_kernel#
stream_3_stream_3_conv2d_3_bias%
!stream_4_stream_4_conv2d_4_kernel#
stream_4_stream_4_conv2d_4_bias%
!stream_5_stream_5_conv2d_5_kernel#
stream_5_stream_5_conv2d_5_bias%
!stream_6_stream_6_conv2d_6_kernel#
stream_6_stream_6_conv2d_6_bias
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’dropout/StatefulPartitionedCall’stream/StatefulPartitionedCall’ stream_1/StatefulPartitionedCall’ stream_2/StatefulPartitionedCall’ stream_3/StatefulPartitionedCall’ stream_4/StatefulPartitionedCall’ stream_5/StatefulPartitionedCall’ stream_6/StatefulPartitionedCallϊ
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3462(
&tf_op_layer_ExpandDims/PartitionedCallΙ
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stream_layer_call_and_return_conditional_losses_3652 
stream/StatefulPartitionedCallΣ
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_1_layer_call_and_return_conditional_losses_3882"
 stream_1/StatefulPartitionedCallΥ
 stream_2/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0!stream_2_stream_2_conv2d_2_kernelstream_2_stream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_2_layer_call_and_return_conditional_losses_4112"
 stream_2/StatefulPartitionedCallΥ
 stream_3/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0!stream_3_stream_3_conv2d_3_kernelstream_3_stream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_3_layer_call_and_return_conditional_losses_4342"
 stream_3/StatefulPartitionedCallΦ
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0!stream_4_stream_4_conv2d_4_kernelstream_4_stream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_4_layer_call_and_return_conditional_losses_4572"
 stream_4/StatefulPartitionedCallΥ
 stream_5/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0!stream_5_stream_5_conv2d_5_kernelstream_5_stream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_5_layer_call_and_return_conditional_losses_4802"
 stream_5/StatefulPartitionedCallΦ
 stream_6/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0!stream_6_stream_6_conv2d_6_kernelstream_6_stream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_6_layer_call_and_return_conditional_losses_5032"
 stream_6/StatefulPartitionedCallμ
stream_7/PartitionedCallPartitionedCall)stream_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_7_layer_call_and_return_conditional_losses_5212
stream_7/PartitionedCallω
dropout/StatefulPartitionedCallStatefulPartitionedCall!stream_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_5422!
dropout/StatefulPartitionedCall¦
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
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_5702
dense/StatefulPartitionedCall²
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_5932!
dense_1/StatefulPartitionedCall³
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_6152!
dense_2/StatefulPartitionedCallμ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dropout/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
½
α
+__inference_functional_1_layer_call_fn_1028
input_1
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1stream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstream_6_conv2d_6_kernelstream_6_conv2d_6_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_functional_1_layer_call_and_return_conditional_losses_7672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
Ή

Ϊ
B__inference_stream_2_layer_call_and_return_conditional_losses_1297

inputs;
7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel:
6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias
identityΐ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΆ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_2/Conv2D΅
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp£
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_2/BiasAddr
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_2/Relun
IdentityIdentityconv2d_2/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Η
]
A__inference_stream_7_layer_call_and_return_conditional_losses_521

inputs
identityo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
flatten/Constw
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*
_output_shapes
:	2
flatten/Reshaped
IdentityIdentityflatten/Reshape:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs


'__inference_stream_2_layer_call_fn_1304

inputs
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_2_conv2d_2_kernelstream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_2_layer_call_and_return_conditional_losses_4112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
½

$__inference_dense_layer_call_fn_1431

inputs
dense_kernel

dense_bias
identity’StatefulPartitionedCallμ
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
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_5702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
³
³
A__inference_dense_2_layer_call_and_return_conditional_losses_1459

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
ό

%__inference_stream_layer_call_fn_1268

inputs
stream_conv2d_kernel
stream_conv2d_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stream_layer_call_and_return_conditional_losses_3652
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:
 
_user_specified_nameinputs
½
α
+__inference_functional_1_layer_call_fn_1003
input_1
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinput_1stream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstream_6_conv2d_6_kernelstream_6_conv2d_6_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_functional_1_layer_call_and_return_conditional_losses_7052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1


'__inference_stream_1_layer_call_fn_1286

inputs
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_1_conv2d_1_kernelstream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_1_layer_call_and_return_conditional_losses_3882
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
λ
B
&__inference_dropout_layer_call_fn_1414

inputs
identityΆ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_5472
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Ί
ΰ
+__inference_functional_1_layer_call_fn_1239

inputs
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstream_6_conv2d_6_kernelstream_6_conv2d_6_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_functional_1_layer_call_and_return_conditional_losses_7672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
χ
_
&__inference_dropout_layer_call_fn_1409

inputs
identity’StatefulPartitionedCallΞ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_5422
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
ϊ?
έ
E__inference_functional_1_layer_call_and_return_conditional_losses_767

inputs
stream_stream_conv2d_kernel
stream_stream_conv2d_bias%
!stream_1_stream_1_conv2d_1_kernel#
stream_1_stream_1_conv2d_1_bias%
!stream_2_stream_2_conv2d_2_kernel#
stream_2_stream_2_conv2d_2_bias%
!stream_3_stream_3_conv2d_3_kernel#
stream_3_stream_3_conv2d_3_bias%
!stream_4_stream_4_conv2d_4_kernel#
stream_4_stream_4_conv2d_4_bias%
!stream_5_stream_5_conv2d_5_kernel#
stream_5_stream_5_conv2d_5_bias%
!stream_6_stream_6_conv2d_6_kernel#
stream_6_stream_6_conv2d_6_bias
dense_dense_kernel
dense_dense_bias
dense_1_dense_1_kernel
dense_1_dense_1_bias
dense_2_dense_2_kernel
dense_2_dense_2_bias
identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCall’stream/StatefulPartitionedCall’ stream_1/StatefulPartitionedCall’ stream_2/StatefulPartitionedCall’ stream_3/StatefulPartitionedCall’ stream_4/StatefulPartitionedCall’ stream_5/StatefulPartitionedCall’ stream_6/StatefulPartitionedCallϊ
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3462(
&tf_op_layer_ExpandDims/PartitionedCallΙ
stream/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0stream_stream_conv2d_kernelstream_stream_conv2d_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_stream_layer_call_and_return_conditional_losses_3652 
stream/StatefulPartitionedCallΣ
 stream_1/StatefulPartitionedCallStatefulPartitionedCall'stream/StatefulPartitionedCall:output:0!stream_1_stream_1_conv2d_1_kernelstream_1_stream_1_conv2d_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_1_layer_call_and_return_conditional_losses_3882"
 stream_1/StatefulPartitionedCallΥ
 stream_2/StatefulPartitionedCallStatefulPartitionedCall)stream_1/StatefulPartitionedCall:output:0!stream_2_stream_2_conv2d_2_kernelstream_2_stream_2_conv2d_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_2_layer_call_and_return_conditional_losses_4112"
 stream_2/StatefulPartitionedCallΥ
 stream_3/StatefulPartitionedCallStatefulPartitionedCall)stream_2/StatefulPartitionedCall:output:0!stream_3_stream_3_conv2d_3_kernelstream_3_stream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_3_layer_call_and_return_conditional_losses_4342"
 stream_3/StatefulPartitionedCallΦ
 stream_4/StatefulPartitionedCallStatefulPartitionedCall)stream_3/StatefulPartitionedCall:output:0!stream_4_stream_4_conv2d_4_kernelstream_4_stream_4_conv2d_4_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_4_layer_call_and_return_conditional_losses_4572"
 stream_4/StatefulPartitionedCallΥ
 stream_5/StatefulPartitionedCallStatefulPartitionedCall)stream_4/StatefulPartitionedCall:output:0!stream_5_stream_5_conv2d_5_kernelstream_5_stream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_5_layer_call_and_return_conditional_losses_4802"
 stream_5/StatefulPartitionedCallΦ
 stream_6/StatefulPartitionedCallStatefulPartitionedCall)stream_5/StatefulPartitionedCall:output:0!stream_6_stream_6_conv2d_6_kernelstream_6_stream_6_conv2d_6_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_6_layer_call_and_return_conditional_losses_5032"
 stream_6/StatefulPartitionedCallμ
stream_7/PartitionedCallPartitionedCall)stream_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_7_layer_call_and_return_conditional_losses_5212
stream_7/PartitionedCallα
dropout/PartitionedCallPartitionedCall!stream_7/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dropout_layer_call_and_return_conditional_losses_5472
dropout/PartitionedCall
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
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_5702
dense/StatefulPartitionedCall²
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_5932!
dense_1/StatefulPartitionedCall³
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_6152!
dense_2/StatefulPartitionedCallΚ
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall^stream/StatefulPartitionedCall!^stream_1/StatefulPartitionedCall!^stream_2/StatefulPartitionedCall!^stream_3/StatefulPartitionedCall!^stream_4/StatefulPartitionedCall!^stream_5/StatefulPartitionedCall!^stream_6/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:::::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2@
stream/StatefulPartitionedCallstream/StatefulPartitionedCall2D
 stream_1/StatefulPartitionedCall stream_1/StatefulPartitionedCall2D
 stream_2/StatefulPartitionedCall stream_2/StatefulPartitionedCall2D
 stream_3/StatefulPartitionedCall stream_3/StatefulPartitionedCall2D
 stream_4/StatefulPartitionedCall stream_4/StatefulPartitionedCall2D
 stream_5/StatefulPartitionedCall stream_5/StatefulPartitionedCall2D
 stream_6/StatefulPartitionedCall stream_6/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
£
^
@__inference_dropout_layer_call_and_return_conditional_losses_547

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Ί
ΰ
+__inference_functional_1_layer_call_fn_1214

inputs
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstream_6_conv2d_6_kernelstream_6_conv2d_6_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_functional_1_layer_call_and_return_conditional_losses_7052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*z
_input_shapesi
g:?????????::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϊ
k
O__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_346

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
:2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
ύ
C
'__inference_stream_7_layer_call_fn_1387

inputs
identity·
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_7_layer_call_and_return_conditional_losses_5212
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Χ

_
@__inference_dropout_layer_call_and_return_conditional_losses_542

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yΆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Έ

Ω
A__inference_stream_3_layer_call_and_return_conditional_losses_434

inputs;
7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel:
6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias
identityΐ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp7conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΆ
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_3/Conv2D΅
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp6conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02!
conv2d_3/BiasAdd/ReadVariableOp£
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_3/BiasAddr
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_3/Relun
IdentityIdentityconv2d_3/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Εc
Γ

E__inference_functional_1_layer_call_and_return_conditional_losses_901
input_1<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_biasD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_biasD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_biasD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_biasD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΝ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/ReluΫ
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpο
stream_2/conv2d_2/Conv2DConv2D$stream_1/conv2d_1/Relu:activations:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2DΠ
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpΗ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/ReluΫ
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpο
stream_3/conv2d_3/Conv2DConv2D$stream_2/conv2d_2/Relu:activations:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2DΠ
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpΗ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Reluά
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpπ
stream_4/conv2d_4/Conv2DConv2D$stream_3/conv2d_3/Relu:activations:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2DΡ
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpΘ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Reluά
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpο
stream_5/conv2d_5/Conv2DConv2D$stream_4/conv2d_4/Relu:activations:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2DΠ
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpΗ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Reluά
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpπ
stream_6/conv2d_6/Conv2DConv2D$stream_5/conv2d_5/Relu:activations:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2DΡ
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpΘ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu
stream_7/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
stream_7/flatten/Const°
stream_7/flatten/ReshapeReshape$stream_6/conv2d_6/Relu:activations:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMul!stream_7/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/dropout/ShapeΔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yΦ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
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
dense_2/BiasAddc
IdentityIdentitydense_2/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^::::::::::::::::::::::T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
ΧT
σ

 __inference__traced_restore_1619
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias%
!assignvariableop_4_dense_2_kernel#
assignvariableop_5_dense_2_bias+
'assignvariableop_6_stream_conv2d_kernel)
%assignvariableop_7_stream_conv2d_bias/
+assignvariableop_8_stream_1_conv2d_1_kernel-
)assignvariableop_9_stream_1_conv2d_1_bias0
,assignvariableop_10_stream_2_conv2d_2_kernel.
*assignvariableop_11_stream_2_conv2d_2_bias0
,assignvariableop_12_stream_3_conv2d_3_kernel.
*assignvariableop_13_stream_3_conv2d_3_bias0
,assignvariableop_14_stream_4_conv2d_4_kernel.
*assignvariableop_15_stream_4_conv2d_4_bias0
,assignvariableop_16_stream_5_conv2d_5_kernel.
*assignvariableop_17_stream_5_conv2d_5_bias0
,assignvariableop_18_stream_6_conv2d_6_kernel.
*assignvariableop_19_stream_6_conv2d_6_bias
identity_21’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9₯
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*±
value§B€B6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesΈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
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

Identity_1’
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

Identity_3€
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

Identity_5€
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¬
AssignVariableOp_6AssignVariableOp'assignvariableop_6_stream_conv2d_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7ͺ
AssignVariableOp_7AssignVariableOp%assignvariableop_7_stream_conv2d_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8°
AssignVariableOp_8AssignVariableOp+assignvariableop_8_stream_1_conv2d_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stream_1_conv2d_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10΄
AssignVariableOp_10AssignVariableOp,assignvariableop_10_stream_2_conv2d_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11²
AssignVariableOp_11AssignVariableOp*assignvariableop_11_stream_2_conv2d_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12΄
AssignVariableOp_12AssignVariableOp,assignvariableop_12_stream_3_conv2d_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13²
AssignVariableOp_13AssignVariableOp*assignvariableop_13_stream_3_conv2d_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14΄
AssignVariableOp_14AssignVariableOp,assignvariableop_14_stream_4_conv2d_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15²
AssignVariableOp_15AssignVariableOp*assignvariableop_15_stream_4_conv2d_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16΄
AssignVariableOp_16AssignVariableOp,assignvariableop_16_stream_5_conv2d_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17²
AssignVariableOp_17AssignVariableOp*assignvariableop_17_stream_5_conv2d_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18΄
AssignVariableOp_18AssignVariableOp,assignvariableop_18_stream_6_conv2d_6_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19²
AssignVariableOp_19AssignVariableOp*assignvariableop_19_stream_6_conv2d_6_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
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
Τn
 
__inference__wrapped_model_336
input_1I
Efunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernelH
Dfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_biasQ
Mfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelP
Lfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasQ
Mfunctional_1_stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelP
Lfunctional_1_stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_biasQ
Mfunctional_1_stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelP
Lfunctional_1_stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_biasQ
Mfunctional_1_stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelP
Lfunctional_1_stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_biasQ
Mfunctional_1_stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelP
Lfunctional_1_stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_biasQ
Mfunctional_1_stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelP
Lfunctional_1_stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias9
5functional_1_dense_matmul_readvariableop_dense_kernel8
4functional_1_dense_biasadd_readvariableop_dense_bias=
9functional_1_dense_1_matmul_readvariableop_dense_1_kernel<
8functional_1_dense_1_biasadd_readvariableop_dense_1_bias=
9functional_1_dense_2_matmul_readvariableop_dense_2_kernel<
8functional_1_dense_2_biasadd_readvariableop_dense_2_bias
identity³
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimτ
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:20
.functional_1/tf_op_layer_ExpandDims/ExpandDimsς
0functional_1/stream/conv2d/Conv2D/ReadVariableOpReadVariableOpEfunctional_1_stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype022
0functional_1/stream/conv2d/Conv2D/ReadVariableOp
!functional_1/stream/conv2d/Conv2DConv2D7functional_1/tf_op_layer_ExpandDims/ExpandDims:output:08functional_1/stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2#
!functional_1/stream/conv2d/Conv2Dη
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpReadVariableOpDfunctional_1_stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype023
1functional_1/stream/conv2d/BiasAdd/ReadVariableOpλ
"functional_1/stream/conv2d/BiasAddBiasAdd*functional_1/stream/conv2d/Conv2D:output:09functional_1/stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2$
"functional_1/stream/conv2d/BiasAdd¨
functional_1/stream/conv2d/ReluRelu+functional_1/stream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2!
functional_1/stream/conv2d/Relu
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype026
4functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp
%functional_1/stream_1/conv2d_1/Conv2DConv2D-functional_1/stream/conv2d/Relu:activations:0<functional_1/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_1/conv2d_1/Conv2Dχ
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype027
5functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOpϋ
&functional_1/stream_1/conv2d_1/BiasAddBiasAdd.functional_1/stream_1/conv2d_1/Conv2D:output:0=functional_1/stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_1/conv2d_1/BiasAdd΄
#functional_1/stream_1/conv2d_1/ReluRelu/functional_1/stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_1/conv2d_1/Relu
4functional_1/stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype026
4functional_1/stream_2/conv2d_2/Conv2D/ReadVariableOp£
%functional_1/stream_2/conv2d_2/Conv2DConv2D1functional_1/stream_1/conv2d_1/Relu:activations:0<functional_1/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_2/conv2d_2/Conv2Dχ
5functional_1/stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype027
5functional_1/stream_2/conv2d_2/BiasAdd/ReadVariableOpϋ
&functional_1/stream_2/conv2d_2/BiasAddBiasAdd.functional_1/stream_2/conv2d_2/Conv2D:output:0=functional_1/stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_2/conv2d_2/BiasAdd΄
#functional_1/stream_2/conv2d_2/ReluRelu/functional_1/stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_2/conv2d_2/Relu
4functional_1/stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype026
4functional_1/stream_3/conv2d_3/Conv2D/ReadVariableOp£
%functional_1/stream_3/conv2d_3/Conv2DConv2D1functional_1/stream_2/conv2d_2/Relu:activations:0<functional_1/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_3/conv2d_3/Conv2Dχ
5functional_1/stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype027
5functional_1/stream_3/conv2d_3/BiasAdd/ReadVariableOpϋ
&functional_1/stream_3/conv2d_3/BiasAddBiasAdd.functional_1/stream_3/conv2d_3/Conv2D:output:0=functional_1/stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_3/conv2d_3/BiasAdd΄
#functional_1/stream_3/conv2d_3/ReluRelu/functional_1/stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_3/conv2d_3/Relu
4functional_1/stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype026
4functional_1/stream_4/conv2d_4/Conv2D/ReadVariableOp€
%functional_1/stream_4/conv2d_4/Conv2DConv2D1functional_1/stream_3/conv2d_3/Relu:activations:0<functional_1/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2'
%functional_1/stream_4/conv2d_4/Conv2Dψ
5functional_1/stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype027
5functional_1/stream_4/conv2d_4/BiasAdd/ReadVariableOpό
&functional_1/stream_4/conv2d_4/BiasAddBiasAdd.functional_1/stream_4/conv2d_4/Conv2D:output:0=functional_1/stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2(
&functional_1/stream_4/conv2d_4/BiasAdd΅
#functional_1/stream_4/conv2d_4/ReluRelu/functional_1/stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2%
#functional_1/stream_4/conv2d_4/Relu
4functional_1/stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype026
4functional_1/stream_5/conv2d_5/Conv2D/ReadVariableOp£
%functional_1/stream_5/conv2d_5/Conv2DConv2D1functional_1/stream_4/conv2d_4/Relu:activations:0<functional_1/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2'
%functional_1/stream_5/conv2d_5/Conv2Dχ
5functional_1/stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype027
5functional_1/stream_5/conv2d_5/BiasAdd/ReadVariableOpϋ
&functional_1/stream_5/conv2d_5/BiasAddBiasAdd.functional_1/stream_5/conv2d_5/Conv2D:output:0=functional_1/stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2(
&functional_1/stream_5/conv2d_5/BiasAdd΄
#functional_1/stream_5/conv2d_5/ReluRelu/functional_1/stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2%
#functional_1/stream_5/conv2d_5/Relu
4functional_1/stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype026
4functional_1/stream_6/conv2d_6/Conv2D/ReadVariableOp€
%functional_1/stream_6/conv2d_6/Conv2DConv2D1functional_1/stream_5/conv2d_5/Relu:activations:0<functional_1/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2'
%functional_1/stream_6/conv2d_6/Conv2Dψ
5functional_1/stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype027
5functional_1/stream_6/conv2d_6/BiasAdd/ReadVariableOpό
&functional_1/stream_6/conv2d_6/BiasAddBiasAdd.functional_1/stream_6/conv2d_6/Conv2D:output:0=functional_1/stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2(
&functional_1/stream_6/conv2d_6/BiasAdd΅
#functional_1/stream_6/conv2d_6/ReluRelu/functional_1/stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2%
#functional_1/stream_6/conv2d_6/Relu
#functional_1/stream_7/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2%
#functional_1/stream_7/flatten/Constδ
%functional_1/stream_7/flatten/ReshapeReshape1functional_1/stream_6/conv2d_6/Relu:activations:0,functional_1/stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2'
%functional_1/stream_7/flatten/Reshape€
functional_1/dropout/IdentityIdentity.functional_1/stream_7/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/IdentityΜ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp5functional_1_dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpΔ
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/MatMulΘ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes	
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpΕ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense/BiasAddΤ
*functional_1/dense_1/MatMul/ReadVariableOpReadVariableOp9functional_1_dense_1_matmul_readvariableop_dense_1_kernel* 
_output_shapes
:
*
dtype02,
*functional_1/dense_1/MatMul/ReadVariableOpΗ
functional_1/dense_1/MatMulMatMul#functional_1/dense/BiasAdd:output:02functional_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/MatMulΠ
+functional_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp8functional_1_dense_1_biasadd_readvariableop_dense_1_bias*
_output_shapes	
:*
dtype02-
+functional_1/dense_1/BiasAdd/ReadVariableOpΝ
functional_1/dense_1/BiasAddBiasAdd%functional_1/dense_1/MatMul:product:03functional_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
functional_1/dense_1/BiasAdd
functional_1/dense_1/ReluRelu%functional_1/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:	2
functional_1/dense_1/ReluΣ
*functional_1/dense_2/MatMul/ReadVariableOpReadVariableOp9functional_1_dense_2_matmul_readvariableop_dense_2_kernel*
_output_shapes
:	*
dtype02,
*functional_1/dense_2/MatMul/ReadVariableOpΚ
functional_1/dense_2/MatMulMatMul'functional_1/dense_1/Relu:activations:02functional_1/dense_2/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/MatMulΟ
+functional_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp8functional_1_dense_2_biasadd_readvariableop_dense_2_bias*
_output_shapes
:*
dtype02-
+functional_1/dense_2/BiasAdd/ReadVariableOpΜ
functional_1/dense_2/BiasAddBiasAdd%functional_1/dense_2/MatMul:product:03functional_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense_2/BiasAddp
IdentityIdentity%functional_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^::::::::::::::::::::::T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1


Λ
?__inference_stream_layer_call_and_return_conditional_losses_365

inputs5
1conv2d_conv2d_readvariableop_stream_conv2d_kernel4
0conv2d_biasadd_readvariableop_stream_conv2d_bias
identityΆ
conv2d/Conv2D/ReadVariableOpReadVariableOp1conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02
conv2d/Conv2D/ReadVariableOp°
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d/Conv2D«
conv2d/BiasAdd/ReadVariableOpReadVariableOp0conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02
conv2d/BiasAdd/ReadVariableOp
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d/BiasAddl
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d/Relul
IdentityIdentityconv2d/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
::::N J
&
_output_shapes
:
 
_user_specified_nameinputs
Έ

Ω
A__inference_stream_1_layer_call_and_return_conditional_losses_388

inputs;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identityΐ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOpΆ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_1/Conv2D΅
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp£
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_1/Relun
IdentityIdentityconv2d_1/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
άZ
Γ

F__inference_functional_1_layer_call_and_return_conditional_losses_1189

inputs<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_biasD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_biasD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_biasD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_biasD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΜ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/ReluΫ
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpο
stream_2/conv2d_2/Conv2DConv2D$stream_1/conv2d_1/Relu:activations:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2DΠ
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpΗ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/ReluΫ
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpο
stream_3/conv2d_3/Conv2DConv2D$stream_2/conv2d_2/Relu:activations:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2DΠ
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpΗ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Reluά
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpπ
stream_4/conv2d_4/Conv2DConv2D$stream_3/conv2d_3/Relu:activations:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2DΡ
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpΘ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Reluά
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpο
stream_5/conv2d_5/Conv2DConv2D$stream_4/conv2d_4/Relu:activations:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2DΠ
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpΗ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Reluά
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpπ
stream_6/conv2d_6/Conv2DConv2D$stream_5/conv2d_5/Relu:activations:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2DΡ
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpΘ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu
stream_7/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
stream_7/flatten/Const°
stream_7/flatten/ReshapeReshape$stream_6/conv2d_6/Relu:activations:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshape}
dropout/IdentityIdentity!stream_7/flatten/Reshape:output:0*
T0*
_output_shapes
:	2
dropout/Identity₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
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
dense_2/BiasAddc
IdentityIdentitydense_2/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^::::::::::::::::::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
ϊ
Χ
!__inference_signature_wrapper_817
input_1
stream_conv2d_kernel
stream_conv2d_bias
stream_1_conv2d_1_kernel
stream_1_conv2d_1_bias
stream_2_conv2d_2_kernel
stream_2_conv2d_2_bias
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
stream_4_conv2d_4_kernel
stream_4_conv2d_4_bias
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
stream_6_conv2d_6_kernel
stream_6_conv2d_6_bias
dense_kernel

dense_bias
dense_1_kernel
dense_1_bias
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1stream_conv2d_kernelstream_conv2d_biasstream_1_conv2d_1_kernelstream_1_conv2d_1_biasstream_2_conv2d_2_kernelstream_2_conv2d_2_biasstream_3_conv2d_3_kernelstream_3_conv2d_3_biasstream_4_conv2d_4_kernelstream_4_conv2d_4_biasstream_5_conv2d_5_kernelstream_5_conv2d_5_biasstream_6_conv2d_6_kernelstream_6_conv2d_6_biasdense_kernel
dense_biasdense_1_kerneldense_1_biasdense_2_kerneldense_2_bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__wrapped_model_3362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^:::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:
!
_user_specified_name	input_1
Ύ

Ω
A__inference_stream_4_layer_call_and_return_conditional_losses_457

inputs;
7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel:
6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias
identityΑ
conv2d_4/Conv2D/ReadVariableOpReadVariableOp7conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp·
conv2d_4/Conv2DConv2Dinputs&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
conv2d_4/Conv2DΆ
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp6conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp€
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
conv2d_4/BiasAdds
conv2d_4/ReluReluconv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
conv2d_4/Reluo
IdentityIdentityconv2d_4/Relu:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ή

Ϊ
B__inference_stream_1_layer_call_and_return_conditional_losses_1279

inputs;
7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel:
6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias
identityΐ
conv2d_1/Conv2D/ReadVariableOpReadVariableOp7conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOpΆ
conv2d_1/Conv2DConv2Dinputs&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_1/Conv2D΅
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp6conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp£
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_1/BiasAddr
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_1/Relun
IdentityIdentityconv2d_1/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Γc
Γ

F__inference_functional_1_layer_call_and_return_conditional_losses_1112

inputs<
8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel;
7stream_conv2d_biasadd_readvariableop_stream_conv2d_biasD
@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernelC
?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_biasD
@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernelC
?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_biasD
@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernelC
?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_biasD
@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernelC
?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_biasD
@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernelC
?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_biasD
@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernelC
?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias0
,dense_1_matmul_readvariableop_dense_1_kernel/
+dense_1_biasadd_readvariableop_dense_1_bias0
,dense_2_matmul_readvariableop_dense_2_kernel/
+dense_2_biasadd_readvariableop_dense_2_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%tf_op_layer_ExpandDims/ExpandDims/dimΜ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:2#
!tf_op_layer_ExpandDims/ExpandDimsΛ
#stream/conv2d/Conv2D/ReadVariableOpReadVariableOp8stream_conv2d_conv2d_readvariableop_stream_conv2d_kernel*&
_output_shapes
:@*
dtype02%
#stream/conv2d/Conv2D/ReadVariableOpι
stream/conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0+stream/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream/conv2d/Conv2Dΐ
$stream/conv2d/BiasAdd/ReadVariableOpReadVariableOp7stream_conv2d_biasadd_readvariableop_stream_conv2d_bias*
_output_shapes
:@*
dtype02&
$stream/conv2d/BiasAdd/ReadVariableOp·
stream/conv2d/BiasAddBiasAddstream/conv2d/Conv2D:output:0,stream/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream/conv2d/BiasAdd
stream/conv2d/ReluRelustream/conv2d/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream/conv2d/ReluΫ
'stream_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@stream_1_conv2d_1_conv2d_readvariableop_stream_1_conv2d_1_kernel*&
_output_shapes
:@@*
dtype02)
'stream_1/conv2d_1/Conv2D/ReadVariableOpλ
stream_1/conv2d_1/Conv2DConv2D stream/conv2d/Relu:activations:0/stream_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_1/conv2d_1/Conv2DΠ
(stream_1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp?stream_1_conv2d_1_biasadd_readvariableop_stream_1_conv2d_1_bias*
_output_shapes
:@*
dtype02*
(stream_1/conv2d_1/BiasAdd/ReadVariableOpΗ
stream_1/conv2d_1/BiasAddBiasAdd!stream_1/conv2d_1/Conv2D:output:00stream_1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/BiasAdd
stream_1/conv2d_1/ReluRelu"stream_1/conv2d_1/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_1/conv2d_1/ReluΫ
'stream_2/conv2d_2/Conv2D/ReadVariableOpReadVariableOp@stream_2_conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02)
'stream_2/conv2d_2/Conv2D/ReadVariableOpο
stream_2/conv2d_2/Conv2DConv2D$stream_1/conv2d_1/Relu:activations:0/stream_2/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_2/conv2d_2/Conv2DΠ
(stream_2/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp?stream_2_conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02*
(stream_2/conv2d_2/BiasAdd/ReadVariableOpΗ
stream_2/conv2d_2/BiasAddBiasAdd!stream_2/conv2d_2/Conv2D:output:00stream_2/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/BiasAdd
stream_2/conv2d_2/ReluRelu"stream_2/conv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_2/conv2d_2/ReluΫ
'stream_3/conv2d_3/Conv2D/ReadVariableOpReadVariableOp@stream_3_conv2d_3_conv2d_readvariableop_stream_3_conv2d_3_kernel*&
_output_shapes
:@@*
dtype02)
'stream_3/conv2d_3/Conv2D/ReadVariableOpο
stream_3/conv2d_3/Conv2DConv2D$stream_2/conv2d_2/Relu:activations:0/stream_3/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_3/conv2d_3/Conv2DΠ
(stream_3/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp?stream_3_conv2d_3_biasadd_readvariableop_stream_3_conv2d_3_bias*
_output_shapes
:@*
dtype02*
(stream_3/conv2d_3/BiasAdd/ReadVariableOpΗ
stream_3/conv2d_3/BiasAddBiasAdd!stream_3/conv2d_3/Conv2D:output:00stream_3/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/BiasAdd
stream_3/conv2d_3/ReluRelu"stream_3/conv2d_3/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_3/conv2d_3/Reluά
'stream_4/conv2d_4/Conv2D/ReadVariableOpReadVariableOp@stream_4_conv2d_4_conv2d_readvariableop_stream_4_conv2d_4_kernel*'
_output_shapes
:@*
dtype02)
'stream_4/conv2d_4/Conv2D/ReadVariableOpπ
stream_4/conv2d_4/Conv2DConv2D$stream_3/conv2d_3/Relu:activations:0/stream_4/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_4/conv2d_4/Conv2DΡ
(stream_4/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp?stream_4_conv2d_4_biasadd_readvariableop_stream_4_conv2d_4_bias*
_output_shapes	
:*
dtype02*
(stream_4/conv2d_4/BiasAdd/ReadVariableOpΘ
stream_4/conv2d_4/BiasAddBiasAdd!stream_4/conv2d_4/Conv2D:output:00stream_4/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/BiasAdd
stream_4/conv2d_4/ReluRelu"stream_4/conv2d_4/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_4/conv2d_4/Reluά
'stream_5/conv2d_5/Conv2D/ReadVariableOpReadVariableOp@stream_5_conv2d_5_conv2d_readvariableop_stream_5_conv2d_5_kernel*'
_output_shapes
:@*
dtype02)
'stream_5/conv2d_5/Conv2D/ReadVariableOpο
stream_5/conv2d_5/Conv2DConv2D$stream_4/conv2d_4/Relu:activations:0/stream_5/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
stream_5/conv2d_5/Conv2DΠ
(stream_5/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp?stream_5_conv2d_5_biasadd_readvariableop_stream_5_conv2d_5_bias*
_output_shapes
:@*
dtype02*
(stream_5/conv2d_5/BiasAdd/ReadVariableOpΗ
stream_5/conv2d_5/BiasAddBiasAdd!stream_5/conv2d_5/Conv2D:output:00stream_5/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/BiasAdd
stream_5/conv2d_5/ReluRelu"stream_5/conv2d_5/BiasAdd:output:0*
T0*&
_output_shapes
:@2
stream_5/conv2d_5/Reluά
'stream_6/conv2d_6/Conv2D/ReadVariableOpReadVariableOp@stream_6_conv2d_6_conv2d_readvariableop_stream_6_conv2d_6_kernel*'
_output_shapes
:@*
dtype02)
'stream_6/conv2d_6/Conv2D/ReadVariableOpπ
stream_6/conv2d_6/Conv2DConv2D$stream_5/conv2d_5/Relu:activations:0/stream_6/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingVALID*
strides
2
stream_6/conv2d_6/Conv2DΡ
(stream_6/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp?stream_6_conv2d_6_biasadd_readvariableop_stream_6_conv2d_6_bias*
_output_shapes	
:*
dtype02*
(stream_6/conv2d_6/BiasAdd/ReadVariableOpΘ
stream_6/conv2d_6/BiasAddBiasAdd!stream_6/conv2d_6/Conv2D:output:00stream_6/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/BiasAdd
stream_6/conv2d_6/ReluRelu"stream_6/conv2d_6/BiasAdd:output:0*
T0*'
_output_shapes
:2
stream_6/conv2d_6/Relu
stream_7/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  2
stream_7/flatten/Const°
stream_7/flatten/ReshapeReshape$stream_6/conv2d_6/Relu:activations:0stream_7/flatten/Const:output:0*
T0*
_output_shapes
:	2
stream_7/flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/dropout/Const
dropout/dropout/MulMul!stream_7/flatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/dropout/ShapeΔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2 
dropout/dropout/GreaterEqual/yΦ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1₯
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes
:	2
dense/MatMul‘
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
dense_2/BiasAddc
IdentityIdentitydense_2/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*q
_input_shapes`
^::::::::::::::::::::::S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ϋ
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_1245

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
:2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
€
_
A__inference_dropout_layer_call_and_return_conditional_losses_1404

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs

²
@__inference_dense_1_layer_call_and_return_conditional_losses_593

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


'__inference_stream_5_layer_call_fn_1358

inputs
stream_5_conv2d_5_kernel
stream_5_conv2d_5_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_5_conv2d_5_kernelstream_5_conv2d_5_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_5_layer_call_and_return_conditional_losses_4802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*.
_input_shapes
:::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ψ

`
A__inference_dropout_layer_call_and_return_conditional_losses_1399

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"     2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/GreaterEqual/yΆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Θ

&__inference_dense_2_layer_call_fn_1466

inputs
dense_2_kernel
dense_2_bias
identity’StatefulPartitionedCallρ
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
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_6152
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
?
­
?__inference_dense_layer_call_and_return_conditional_losses_1424

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel* 
_output_shapes
:
*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs

Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_1250

inputs
identityΜ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3462
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:2

Identity"
identityIdentity:output:0*!
_input_shapes
::J F
"
_output_shapes
:
 
_user_specified_nameinputs
Έ

Ω
A__inference_stream_2_layer_call_and_return_conditional_losses_411

inputs;
7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel:
6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias
identityΐ
conv2d_2/Conv2D/ReadVariableOpReadVariableOp7conv2d_2_conv2d_readvariableop_stream_2_conv2d_2_kernel*&
_output_shapes
:@@*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΆ
conv2d_2/Conv2DConv2Dinputs&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingVALID*
strides
2
conv2d_2/Conv2D΅
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp6conv2d_2_biasadd_readvariableop_stream_2_conv2d_2_bias*
_output_shapes
:@*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp£
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2
conv2d_2/BiasAddr
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*&
_output_shapes
:@2
conv2d_2/Relun
IdentityIdentityconv2d_2/Relu:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@:::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
²
²
@__inference_dense_2_layer_call_and_return_conditional_losses_615

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


'__inference_stream_3_layer_call_fn_1322

inputs
stream_3_conv2d_3_kernel
stream_3_conv2d_3_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsstream_3_conv2d_3_kernelstream_3_conv2d_3_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_stream_3_layer_call_and_return_conditional_losses_4342
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*-
_input_shapes
:@::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs"ΈL
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
serving_default_input_1:02
dense_2'
StatefulPartitionedCall:0tensorflow/serving/predict:ΊΜ
€
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api

signatures
ϊ_default_save_signature
ϋ__call__
+ό&call_and_return_all_conditional_losses"
_tf_keras_networkψ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 25, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 20, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 18, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 16, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_2", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_3", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 14, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_3", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_4", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 12, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_4", "inbound_nodes": [[["stream_3", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_5", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 11, 128], "ring_buffer_size_in_time_dim": 5}, "name": "stream_5", "inbound_nodes": [[["stream_4", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_6", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 11, 64], "ring_buffer_size_in_time_dim": 3}, "name": "stream_6", "inbound_nodes": [[["stream_5", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_7", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 11, 128], "ring_buffer_size_in_time_dim": null}, "name": "stream_7", "inbound_nodes": [[["stream_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 25, 20]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 25, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 20, 1], "ring_buffer_size_in_time_dim": 3}, "name": "stream", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 18, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_1", "inbound_nodes": [[["stream", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 16, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_2", "inbound_nodes": [[["stream_1", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_3", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 14, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_3", "inbound_nodes": [[["stream_2", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_4", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 12, 64], "ring_buffer_size_in_time_dim": 5}, "name": "stream_4", "inbound_nodes": [[["stream_3", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_5", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 11, 128], "ring_buffer_size_in_time_dim": 5}, "name": "stream_5", "inbound_nodes": [[["stream_4", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_6", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 11, 64], "ring_buffer_size_in_time_dim": 3}, "name": "stream_6", "inbound_nodes": [[["stream_5", 0, 0, {}]]]}, {"class_name": "Stream", "config": {"name": "stream_7", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 11, 128], "ring_buffer_size_in_time_dim": null}, "name": "stream_7", "inbound_nodes": [[["stream_6", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["stream_7", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_2", 0, 0]]}}}
ν"κ
_tf_keras_input_layerΚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 25, 20]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 25, 20]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

	variables
trainable_variables
regularization_losses
	keras_api
ύ__call__
+ώ&call_and_return_all_conditional_losses"ϊ
_tf_keras_layerΰ{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": -1}}}
λ

cell
state_shape
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+&call_and_return_all_conditional_losses"Ώ	
_tf_keras_layer₯	{"class_name": "Stream", "name": "stream", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 20, 1], "ring_buffer_size_in_time_dim": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 25, 20, 1]}}
σ

cell
state_shape
 	variables
!trainable_variables
"regularization_losses
#	keras_api
__call__
+&call_and_return_all_conditional_losses"Η	
_tf_keras_layer­	{"class_name": "Stream", "name": "stream_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_1", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 18, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 23, 18, 64]}}
σ

$cell
%state_shape
&	variables
'trainable_variables
(regularization_losses
)	keras_api
__call__
+&call_and_return_all_conditional_losses"Η	
_tf_keras_layer­	{"class_name": "Stream", "name": "stream_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_2", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 16, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 19, 16, 64]}}
σ

*cell
+state_shape
,	variables
-trainable_variables
.regularization_losses
/	keras_api
__call__
+&call_and_return_all_conditional_losses"Η	
_tf_keras_layer­	{"class_name": "Stream", "name": "stream_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_3", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 14, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 15, 14, 64]}}
τ

0cell
1state_shape
2	variables
3trainable_variables
4regularization_losses
5	keras_api
__call__
+&call_and_return_all_conditional_losses"Θ	
_tf_keras_layer?	{"class_name": "Stream", "name": "stream_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_4", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 12, 64], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 11, 12, 64]}}
τ

6cell
7state_shape
8	variables
9trainable_variables
:regularization_losses
;	keras_api
__call__
+&call_and_return_all_conditional_losses"Θ	
_tf_keras_layer?	{"class_name": "Stream", "name": "stream_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_5", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 5, 11, 128], "ring_buffer_size_in_time_dim": 5}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 7, 11, 128]}}
σ

<cell
=state_shape
>	variables
?trainable_variables
@regularization_losses
A	keras_api
__call__
+&call_and_return_all_conditional_losses"Η	
_tf_keras_layer­	{"class_name": "Stream", "name": "stream_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_6", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "__passive_serialization__": true}, "state_shape": [1, 3, 11, 64], "ring_buffer_size_in_time_dim": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3, 11, 64]}}
Φ
Bcell
Cstate_shape
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
__call__
+&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "Stream", "name": "stream_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "stream_7", "trainable": true, "dtype": "float32", "inference_batch_size": 1, "mode": "TRAINING", "pad_time_dim": false, "cell": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "__passive_serialization__": true}, "state_shape": [1, 1, 11, 128], "ring_buffer_size_in_time_dim": null}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 1, 11, 128]}}
γ
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.5, "noise_shape": null, "seed": null}}
ͺ

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerι{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1408}}}}
«

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerκ{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 256, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}
¬

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerλ{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}}
Ά
^0
_1
`2
a3
b4
c5
d6
e7
f8
g9
h10
i11
j12
k13
L14
M15
R16
S17
X18
Y19"
trackable_list_wrapper
Ά
^0
_1
`2
a3
b4
c5
d6
e7
f8
g9
h10
i11
j12
k13
L14
M15
R16
S17
X18
Y19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ξ
	variables
lnon_trainable_variables
mmetrics
nlayer_regularization_losses

olayers
player_metrics
trainable_variables
regularization_losses
ϋ__call__
ϊ_default_save_signature
+ό&call_and_return_all_conditional_losses
'ό"call_and_return_conditional_losses"
_generic_user_object
-
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
°
qnon_trainable_variables
	variables
rlayer_metrics

slayers
trainable_variables
tlayer_regularization_losses
umetrics
regularization_losses
ύ__call__
+ώ&call_and_return_all_conditional_losses
'ώ"call_and_return_conditional_losses"
_generic_user_object
	

^kernel
_bias
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
__call__
+&call_and_return_all_conditional_losses"ψ
_tf_keras_layerή{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}}
 "
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
°
znon_trainable_variables
	variables
{layer_metrics

|layers
trainable_variables
}layer_regularization_losses
~metrics
regularization_losses
?__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
§	

`kernel
abias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ύ
_tf_keras_layerγ{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
non_trainable_variables
 	variables
layer_metrics
layers
!trainable_variables
 layer_regularization_losses
metrics
"regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨	

bkernel
cbias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ύ
_tf_keras_layerγ{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
non_trainable_variables
&	variables
layer_metrics
layers
'trainable_variables
 layer_regularization_losses
metrics
(regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
¨	

dkernel
ebias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"ύ
_tf_keras_layerγ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
non_trainable_variables
,	variables
layer_metrics
layers
-trainable_variables
 layer_regularization_losses
metrics
.regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
©	

fkernel
gbias
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
+‘&call_and_return_all_conditional_losses"ώ
_tf_keras_layerδ{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [5, 2]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
non_trainable_variables
2	variables
layer_metrics
 layers
3trainable_variables
 ‘layer_regularization_losses
’metrics
4regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
©	

hkernel
ibias
£	variables
€trainable_variables
₯regularization_losses
¦	keras_api
’__call__
+£&call_and_return_all_conditional_losses"ώ
_tf_keras_layerδ{"class_name": "Conv2D", "name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}}
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
§non_trainable_variables
8	variables
¨layer_metrics
©layers
9trainable_variables
 ͺlayer_regularization_losses
«metrics
:regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
©	

jkernel
kbias
¬	variables
­trainable_variables
?regularization_losses
―	keras_api
€__call__
+₯&call_and_return_all_conditional_losses"ώ
_tf_keras_layerδ{"class_name": "Conv2D", "name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
°non_trainable_variables
>	variables
±layer_metrics
²layers
?trainable_variables
 ³layer_regularization_losses
΄metrics
@regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
θ
΅	variables
Άtrainable_variables
·regularization_losses
Έ	keras_api
¦__call__
+§&call_and_return_all_conditional_losses"Σ
_tf_keras_layerΉ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Ήnon_trainable_variables
D	variables
Ίlayer_metrics
»layers
Etrainable_variables
 Όlayer_regularization_losses
½metrics
Fregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Ύnon_trainable_variables
H	variables
Ώlayer_metrics
ΐlayers
Itrainable_variables
 Αlayer_regularization_losses
Βmetrics
Jregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 :
2dense/kernel
:2
dense/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Γnon_trainable_variables
N	variables
Δlayer_metrics
Εlayers
Otrainable_variables
 Ζlayer_regularization_losses
Ηmetrics
Pregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Θnon_trainable_variables
T	variables
Ιlayer_metrics
Κlayers
Utrainable_variables
 Λlayer_regularization_losses
Μmetrics
Vregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
!:	2dense_2/kernel
:2dense_2/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
Νnon_trainable_variables
Z	variables
Ξlayer_metrics
Οlayers
[trainable_variables
 Πlayer_regularization_losses
Ρmetrics
\regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
.:,@2stream/conv2d/kernel
 :@2stream/conv2d/bias
2:0@@2stream_1/conv2d_1/kernel
$:"@2stream_1/conv2d_1/bias
2:0@@2stream_2/conv2d_2/kernel
$:"@2stream_2/conv2d_2/bias
2:0@@2stream_3/conv2d_3/kernel
$:"@2stream_3/conv2d_3/bias
3:1@2stream_4/conv2d_4/kernel
%:#2stream_4/conv2d_4/bias
3:1@2stream_5/conv2d_5/kernel
$:"@2stream_5/conv2d_5/bias
3:1@2stream_6/conv2d_6/kernel
%:#2stream_6/conv2d_6/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper

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
11
12
13"
trackable_list_wrapper
 "
trackable_dict_wrapper
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
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
?non_trainable_variables
v	variables
Σlayer_metrics
Τlayers
wtrainable_variables
 Υlayer_regularization_losses
Φmetrics
xregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
Χnon_trainable_variables
	variables
Ψlayer_metrics
Ωlayers
trainable_variables
 Ϊlayer_regularization_losses
Ϋmetrics
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
άnon_trainable_variables
	variables
έlayer_metrics
ήlayers
trainable_variables
 ίlayer_regularization_losses
ΰmetrics
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
$0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
αnon_trainable_variables
	variables
βlayer_metrics
γlayers
trainable_variables
 δlayer_regularization_losses
εmetrics
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
.
f0
g1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ζnon_trainable_variables
	variables
ηlayer_metrics
θlayers
trainable_variables
 ιlayer_regularization_losses
κmetrics
regularization_losses
 __call__
+‘&call_and_return_all_conditional_losses
'‘"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
00"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
λnon_trainable_variables
£	variables
μlayer_metrics
νlayers
€trainable_variables
 ξlayer_regularization_losses
οmetrics
₯regularization_losses
’__call__
+£&call_and_return_all_conditional_losses
'£"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
60"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
πnon_trainable_variables
¬	variables
ρlayer_metrics
ςlayers
­trainable_variables
 σlayer_regularization_losses
τmetrics
?regularization_losses
€__call__
+₯&call_and_return_all_conditional_losses
'₯"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
<0"
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
Έ
υnon_trainable_variables
΅	variables
φlayer_metrics
χlayers
Άtrainable_variables
 ψlayer_regularization_losses
ωmetrics
·regularization_losses
¦__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
B0"
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
ΰ2έ
__inference__wrapped_model_336Ί
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
annotationsͺ **’'
%"
input_1?????????
ϊ2χ
+__inference_functional_1_layer_call_fn_1003
+__inference_functional_1_layer_call_fn_1028
+__inference_functional_1_layer_call_fn_1214
+__inference_functional_1_layer_call_fn_1239ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
δ2α
F__inference_functional_1_layer_call_and_return_conditional_losses_1189
E__inference_functional_1_layer_call_and_return_conditional_losses_901
E__inference_functional_1_layer_call_and_return_conditional_losses_978
F__inference_functional_1_layer_call_and_return_conditional_losses_1112ΐ
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
kwonlydefaultsͺ 
annotationsͺ *
 
ί2ά
5__inference_tf_op_layer_ExpandDims_layer_call_fn_1250’
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
annotationsͺ *
 
ϊ2χ
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_1245’
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
annotationsͺ *
 
Ο2Μ
%__inference_stream_layer_call_fn_1268’
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
annotationsͺ *
 
κ2η
@__inference_stream_layer_call_and_return_conditional_losses_1261’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_1_layer_call_fn_1286’
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
annotationsͺ *
 
μ2ι
B__inference_stream_1_layer_call_and_return_conditional_losses_1279’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_2_layer_call_fn_1304’
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
annotationsͺ *
 
μ2ι
B__inference_stream_2_layer_call_and_return_conditional_losses_1297’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_3_layer_call_fn_1322’
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
annotationsͺ *
 
μ2ι
B__inference_stream_3_layer_call_and_return_conditional_losses_1315’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_4_layer_call_fn_1340’
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
annotationsͺ *
 
μ2ι
B__inference_stream_4_layer_call_and_return_conditional_losses_1333’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_5_layer_call_fn_1358’
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
annotationsͺ *
 
μ2ι
B__inference_stream_5_layer_call_and_return_conditional_losses_1351’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_6_layer_call_fn_1376’
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
annotationsͺ *
 
μ2ι
B__inference_stream_6_layer_call_and_return_conditional_losses_1369’
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
annotationsͺ *
 
Ρ2Ξ
'__inference_stream_7_layer_call_fn_1387’
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
annotationsͺ *
 
μ2ι
B__inference_stream_7_layer_call_and_return_conditional_losses_1382’
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
annotationsͺ *
 
2
&__inference_dropout_layer_call_fn_1409
&__inference_dropout_layer_call_fn_1414΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
ΐ2½
A__inference_dropout_layer_call_and_return_conditional_losses_1404
A__inference_dropout_layer_call_and_return_conditional_losses_1399΄
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
kwonlydefaultsͺ 
annotationsͺ *
 
Ξ2Λ
$__inference_dense_layer_call_fn_1431’
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
annotationsͺ *
 
ι2ζ
?__inference_dense_layer_call_and_return_conditional_losses_1424’
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
annotationsͺ *
 
Π2Ν
&__inference_dense_1_layer_call_fn_1449’
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
annotationsͺ *
 
λ2θ
A__inference_dense_1_layer_call_and_return_conditional_losses_1442’
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
annotationsͺ *
 
Π2Ν
&__inference_dense_2_layer_call_fn_1466’
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
annotationsͺ *
 
λ2θ
A__inference_dense_2_layer_call_and_return_conditional_losses_1459’
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
annotationsͺ *
 
0B.
!__inference_signature_wrapper_817input_1
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
¨2₯’
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
annotationsͺ *
 
__inference__wrapped_model_336v^_`abcdefghijkLMRSXY4’1
*’'
%"
input_1?????????
ͺ "(ͺ%
#
dense_2
dense_2
A__inference_dense_1_layer_call_and_return_conditional_losses_1442LRS'’$
’

inputs	
ͺ "’

0	
 i
&__inference_dense_1_layer_call_fn_1449?RS'’$
’

inputs	
ͺ "	
A__inference_dense_2_layer_call_and_return_conditional_losses_1459KXY'’$
’

inputs	
ͺ "’

0
 h
&__inference_dense_2_layer_call_fn_1466>XY'’$
’

inputs	
ͺ "
?__inference_dense_layer_call_and_return_conditional_losses_1424LLM'’$
’

inputs	
ͺ "’

0	
 g
$__inference_dense_layer_call_fn_1431?LM'’$
’

inputs	
ͺ "	
A__inference_dropout_layer_call_and_return_conditional_losses_1399L+’(
!’

inputs	
p
ͺ "’

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_1404L+’(
!’

inputs	
p 
ͺ "’

0	
 i
&__inference_dropout_layer_call_fn_1409?+’(
!’

inputs	
p
ͺ "	i
&__inference_dropout_layer_call_fn_1414?+’(
!’

inputs	
p 
ͺ "	»
F__inference_functional_1_layer_call_and_return_conditional_losses_1112q^_`abcdefghijkLMRSXY;’8
1’.
$!
inputs?????????
p

 
ͺ "’

0
 »
F__inference_functional_1_layer_call_and_return_conditional_losses_1189q^_`abcdefghijkLMRSXY;’8
1’.
$!
inputs?????????
p 

 
ͺ "’

0
 »
E__inference_functional_1_layer_call_and_return_conditional_losses_901r^_`abcdefghijkLMRSXY<’9
2’/
%"
input_1?????????
p

 
ͺ "’

0
 »
E__inference_functional_1_layer_call_and_return_conditional_losses_978r^_`abcdefghijkLMRSXY<’9
2’/
%"
input_1?????????
p 

 
ͺ "’

0
 
+__inference_functional_1_layer_call_fn_1003e^_`abcdefghijkLMRSXY<’9
2’/
%"
input_1?????????
p

 
ͺ "
+__inference_functional_1_layer_call_fn_1028e^_`abcdefghijkLMRSXY<’9
2’/
%"
input_1?????????
p 

 
ͺ "
+__inference_functional_1_layer_call_fn_1214d^_`abcdefghijkLMRSXY;’8
1’.
$!
inputs?????????
p

 
ͺ "
+__inference_functional_1_layer_call_fn_1239d^_`abcdefghijkLMRSXY;’8
1’.
$!
inputs?????????
p 

 
ͺ "
!__inference_signature_wrapper_817x^_`abcdefghijkLMRSXY6’3
’ 
,ͺ)
'
input_1
input_1"(ͺ%
#
dense_2
dense_2 
B__inference_stream_1_layer_call_and_return_conditional_losses_1279Z`a.’+
$’!

inputs@
ͺ "$’!

0@
 x
'__inference_stream_1_layer_call_fn_1286M`a.’+
$’!

inputs@
ͺ "@ 
B__inference_stream_2_layer_call_and_return_conditional_losses_1297Zbc.’+
$’!

inputs@
ͺ "$’!

0@
 x
'__inference_stream_2_layer_call_fn_1304Mbc.’+
$’!

inputs@
ͺ "@ 
B__inference_stream_3_layer_call_and_return_conditional_losses_1315Zde.’+
$’!

inputs@
ͺ "$’!

0@
 x
'__inference_stream_3_layer_call_fn_1322Mde.’+
$’!

inputs@
ͺ "@‘
B__inference_stream_4_layer_call_and_return_conditional_losses_1333[fg.’+
$’!

inputs@
ͺ "%’"

0
 y
'__inference_stream_4_layer_call_fn_1340Nfg.’+
$’!

inputs@
ͺ "‘
B__inference_stream_5_layer_call_and_return_conditional_losses_1351[hi/’,
%’"
 
inputs
ͺ "$’!

0@
 y
'__inference_stream_5_layer_call_fn_1358Nhi/’,
%’"
 
inputs
ͺ "@‘
B__inference_stream_6_layer_call_and_return_conditional_losses_1369[jk.’+
$’!

inputs@
ͺ "%’"

0
 y
'__inference_stream_6_layer_call_fn_1376Njk.’+
$’!

inputs@
ͺ "
B__inference_stream_7_layer_call_and_return_conditional_losses_1382P/’,
%’"
 
inputs
ͺ "’

0	
 n
'__inference_stream_7_layer_call_fn_1387C/’,
%’"
 
inputs
ͺ "	
@__inference_stream_layer_call_and_return_conditional_losses_1261Z^_.’+
$’!

inputs
ͺ "$’!

0@
 v
%__inference_stream_layer_call_fn_1268M^_.’+
$’!

inputs
ͺ "@¦
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_1245R*’'
 ’

inputs
ͺ "$’!

0
 ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_1250E*’'
 ’

inputs
ͺ "