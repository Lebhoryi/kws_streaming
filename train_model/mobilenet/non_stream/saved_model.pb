ρα(
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
 "serve*2.3.02v2.3.0-rc2-23-gb36436b0878ώι 
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:( *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:( *
dtype0

batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma

-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0

batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta

,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0

batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean

3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0

#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance

7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
¦
!depthwise_conv2d/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!depthwise_conv2d/depthwise_kernel

5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOpReadVariableOp!depthwise_conv2d/depthwise_kernel*&
_output_shapes
: *
dtype0

batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma

/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0

batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta

.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0

!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean

5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
’
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance

9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  * 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:  *
dtype0

batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_2/gamma

/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
: *
dtype0

batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_2/beta

.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
: *
dtype0

!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_2/moving_mean

5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
: *
dtype0
’
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_2/moving_variance

9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
: *
dtype0
ͺ
#depthwise_conv2d_1/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#depthwise_conv2d_1/depthwise_kernel
£
7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_1/depthwise_kernel*&
_output_shapes
: *
dtype0

batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_3/gamma

/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
: *
dtype0

batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_3/beta

.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
: *
dtype0

!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_3/moving_mean

5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
: *
dtype0
’
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_3/moving_variance

9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
: *
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: @*
dtype0

batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma

/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0

batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta

.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0

!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean

5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
’
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance

9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
ͺ
#depthwise_conv2d_2/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#depthwise_conv2d_2/depthwise_kernel
£
7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_2/depthwise_kernel*&
_output_shapes
:@*
dtype0

batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma

/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0

batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta

.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0

!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean

5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
’
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance

9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@* 
shared_nameconv2d_3/kernel
|
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*'
_output_shapes
:@*
dtype0

batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_6/gamma

/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes	
:*
dtype0

batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_6/beta

.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes	
:*
dtype0

!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_6/moving_mean

5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_6/moving_variance

9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes	
:*
dtype0
«
#depthwise_conv2d_3/depthwise_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#depthwise_conv2d_3/depthwise_kernel
€
7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOpReadVariableOp#depthwise_conv2d_3/depthwise_kernel*'
_output_shapes
:*
dtype0

batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_7/gamma

/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes	
:*
dtype0

batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_7/beta

.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes	
:*
dtype0

!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_7/moving_mean

5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_7/moving_variance

9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes	
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
}
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*(
_output_shapes
:*
dtype0

batch_normalization_8/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_8/gamma

/batch_normalization_8/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_8/gamma*
_output_shapes	
:*
dtype0

batch_normalization_8/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namebatch_normalization_8/beta

.batch_normalization_8/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_8/beta*
_output_shapes	
:*
dtype0

!batch_normalization_8/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!batch_normalization_8/moving_mean

5batch_normalization_8/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_8/moving_mean*
_output_shapes	
:*
dtype0
£
%batch_normalization_8/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%batch_normalization_8/moving_variance

9batch_normalization_8/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_8/moving_variance*
_output_shapes	
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Φ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB Bω

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
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-18
 layer-31
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
 
R
&trainable_variables
'	variables
(regularization_losses
)	keras_api
^

*kernel
+trainable_variables
,	variables
-regularization_losses
.	keras_api

/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
R
8trainable_variables
9	variables
:regularization_losses
;	keras_api
h
<depthwise_kernel
=trainable_variables
>	variables
?regularization_losses
@	keras_api

Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
R
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
^

Nkernel
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api

Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
R
\trainable_variables
]	variables
^regularization_losses
_	keras_api
h
`depthwise_kernel
atrainable_variables
b	variables
cregularization_losses
d	keras_api

eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
R
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
^

rkernel
strainable_variables
t	variables
uregularization_losses
v	keras_api

waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}	variables
~regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
m
depthwise_kernel
trainable_variables
	variables
regularization_losses
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
V
trainable_variables
	variables
regularization_losses
	keras_api
c
kernel
trainable_variables
	variables
regularization_losses
	keras_api
 
	axis

gamma
	beta
moving_mean
moving_variance
 trainable_variables
‘	variables
’regularization_losses
£	keras_api
V
€trainable_variables
₯	variables
¦regularization_losses
§	keras_api
m
¨depthwise_kernel
©trainable_variables
ͺ	variables
«regularization_losses
¬	keras_api
 
	­axis

?gamma
	―beta
°moving_mean
±moving_variance
²trainable_variables
³	variables
΄regularization_losses
΅	keras_api
V
Άtrainable_variables
·	variables
Έregularization_losses
Ή	keras_api
c
Ίkernel
»trainable_variables
Ό	variables
½regularization_losses
Ύ	keras_api
 
	Ώaxis

ΐgamma
	Αbeta
Βmoving_mean
Γmoving_variance
Δtrainable_variables
Ε	variables
Ζregularization_losses
Η	keras_api
V
Θtrainable_variables
Ι	variables
Κregularization_losses
Λ	keras_api
V
Μtrainable_variables
Ν	variables
Ξregularization_losses
Ο	keras_api
V
Πtrainable_variables
Ρ	variables
?regularization_losses
Σ	keras_api
n
Τkernel
	Υbias
Φtrainable_variables
Χ	variables
Ψregularization_losses
Ω	keras_api
μ
*0
01
12
<3
B4
C5
N6
T7
U8
`9
f10
g11
r12
x13
y14
15
16
17
18
19
20
¨21
?22
―23
Ί24
ΐ25
Α26
Τ27
Υ28

*0
01
12
23
34
<5
B6
C7
D8
E9
N10
T11
U12
V13
W14
`15
f16
g17
h18
i19
r20
x21
y22
z23
{24
25
26
27
28
29
30
31
32
33
34
¨35
?36
―37
°38
±39
Ί40
ΐ41
Α42
Β43
Γ44
Τ45
Υ46
 
²
Ϊmetrics
Ϋlayers
!trainable_variables
άlayer_metrics
έnon_trainable_variables
 ήlayer_regularization_losses
"	variables
#regularization_losses
 
 
 
 
²
ίmetrics
ΰlayers
&trainable_variables
αlayer_metrics
βnon_trainable_variables
 γlayer_regularization_losses
'	variables
(regularization_losses
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE

*0

*0
 
²
δmetrics
εlayers
+trainable_variables
ζlayer_metrics
ηnon_trainable_variables
 θlayer_regularization_losses
,	variables
-regularization_losses
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

00
11

00
11
22
33
 
²
ιmetrics
κlayers
4trainable_variables
λlayer_metrics
μnon_trainable_variables
 νlayer_regularization_losses
5	variables
6regularization_losses
 
 
 
²
ξmetrics
οlayers
8trainable_variables
πlayer_metrics
ρnon_trainable_variables
 ςlayer_regularization_losses
9	variables
:regularization_losses
wu
VARIABLE_VALUE!depthwise_conv2d/depthwise_kernel@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

<0

<0
 
²
σmetrics
τlayers
=trainable_variables
υlayer_metrics
φnon_trainable_variables
 χlayer_regularization_losses
>	variables
?regularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

B0
C1

B0
C1
D2
E3
 
²
ψmetrics
ωlayers
Ftrainable_variables
ϊlayer_metrics
ϋnon_trainable_variables
 όlayer_regularization_losses
G	variables
Hregularization_losses
 
 
 
²
ύmetrics
ώlayers
Jtrainable_variables
?layer_metrics
non_trainable_variables
 layer_regularization_losses
K	variables
Lregularization_losses
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE

N0

N0
 
²
metrics
layers
Otrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
P	variables
Qregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
V2
W3
 
²
metrics
layers
Xtrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
Y	variables
Zregularization_losses
 
 
 
²
metrics
layers
\trainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
]	variables
^regularization_losses
yw
VARIABLE_VALUE#depthwise_conv2d_1/depthwise_kernel@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

`0

`0
 
²
metrics
layers
atrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
b	variables
cregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

f0
g1

f0
g1
h2
i3
 
²
metrics
layers
jtrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
k	variables
lregularization_losses
 
 
 
²
metrics
layers
ntrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
o	variables
pregularization_losses
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE

r0

r0
 
²
 metrics
‘layers
strainable_variables
’layer_metrics
£non_trainable_variables
 €layer_regularization_losses
t	variables
uregularization_losses
 
fd
VARIABLE_VALUEbatch_normalization_4/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_4/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_4/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_4/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

x0
y1

x0
y1
z2
{3
 
²
₯metrics
¦layers
|trainable_variables
§layer_metrics
¨non_trainable_variables
 ©layer_regularization_losses
}	variables
~regularization_losses
 
 
 
΅
ͺmetrics
«layers
trainable_variables
¬layer_metrics
­non_trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
zx
VARIABLE_VALUE#depthwise_conv2d_2/depthwise_kernelAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
΅
―metrics
°layers
trainable_variables
±layer_metrics
²non_trainable_variables
 ³layer_regularization_losses
	variables
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
0
1
2
3
 
΅
΄metrics
΅layers
trainable_variables
Άlayer_metrics
·non_trainable_variables
 Έlayer_regularization_losses
	variables
regularization_losses
 
 
 
΅
Ήmetrics
Ίlayers
trainable_variables
»layer_metrics
Όnon_trainable_variables
 ½layer_regularization_losses
	variables
regularization_losses
\Z
VARIABLE_VALUEconv2d_3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE

0

0
 
΅
Ύmetrics
Ώlayers
trainable_variables
ΐlayer_metrics
Αnon_trainable_variables
 Βlayer_regularization_losses
	variables
regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
0
1
2
3
 
΅
Γmetrics
Δlayers
 trainable_variables
Εlayer_metrics
Ζnon_trainable_variables
 Ηlayer_regularization_losses
‘	variables
’regularization_losses
 
 
 
΅
Θmetrics
Ιlayers
€trainable_variables
Κlayer_metrics
Λnon_trainable_variables
 Μlayer_regularization_losses
₯	variables
¦regularization_losses
zx
VARIABLE_VALUE#depthwise_conv2d_3/depthwise_kernelAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUE

¨0

¨0
 
΅
Νmetrics
Ξlayers
©trainable_variables
Οlayer_metrics
Πnon_trainable_variables
 Ρlayer_regularization_losses
ͺ	variables
«regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

?0
―1
 
?0
―1
°2
±3
 
΅
?metrics
Σlayers
²trainable_variables
Τlayer_metrics
Υnon_trainable_variables
 Φlayer_regularization_losses
³	variables
΄regularization_losses
 
 
 
΅
Χmetrics
Ψlayers
Άtrainable_variables
Ωlayer_metrics
Ϊnon_trainable_variables
 Ϋlayer_regularization_losses
·	variables
Έregularization_losses
\Z
VARIABLE_VALUEconv2d_4/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE

Ί0

Ί0
 
΅
άmetrics
έlayers
»trainable_variables
ήlayer_metrics
ίnon_trainable_variables
 ΰlayer_regularization_losses
Ό	variables
½regularization_losses
 
ge
VARIABLE_VALUEbatch_normalization_8/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_8/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_8/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_8/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

ΐ0
Α1
 
ΐ0
Α1
Β2
Γ3
 
΅
αmetrics
βlayers
Δtrainable_variables
γlayer_metrics
δnon_trainable_variables
 εlayer_regularization_losses
Ε	variables
Ζregularization_losses
 
 
 
΅
ζmetrics
ηlayers
Θtrainable_variables
θlayer_metrics
ιnon_trainable_variables
 κlayer_regularization_losses
Ι	variables
Κregularization_losses
 
 
 
΅
λmetrics
μlayers
Μtrainable_variables
νlayer_metrics
ξnon_trainable_variables
 οlayer_regularization_losses
Ν	variables
Ξregularization_losses
 
 
 
΅
πmetrics
ρlayers
Πtrainable_variables
ςlayer_metrics
σnon_trainable_variables
 τlayer_regularization_losses
Ρ	variables
?regularization_losses
YW
VARIABLE_VALUEdense/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUE
dense/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

Τ0
Υ1

Τ0
Υ1
 
΅
υmetrics
φlayers
Φtrainable_variables
χlayer_metrics
ψnon_trainable_variables
 ωlayer_regularization_losses
Χ	variables
Ψregularization_losses
 
φ
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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
 

20
31
D2
E3
V4
W5
h6
i7
z8
{9
10
11
12
13
°14
±15
Β16
Γ17
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

20
31
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

D0
E1
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

V0
W1
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

h0
i1
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

z0
{1
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

0
1
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

0
1
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

°0
±1
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

Β0
Γ1
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
:1(*
dtype0*
shape:1(

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!depthwise_conv2d/depthwise_kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#depthwise_conv2d_1/depthwise_kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_2/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_2/depthwise_kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#depthwise_conv2d_3/depthwise_kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_4/kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense/kernel
dense/bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_2913
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp5depthwise_conv2d/depthwise_kernel/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp7depthwise_conv2d_1/depthwise_kernel/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp7depthwise_conv2d_2/depthwise_kernel/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp7depthwise_conv2d_3/depthwise_kernel/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp/batch_normalization_8/gamma/Read/ReadVariableOp.batch_normalization_8/beta/Read/ReadVariableOp5batch_normalization_8/moving_mean/Read/ReadVariableOp9batch_normalization_8/moving_variance/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpConst*<
Tin5
321*
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
__inference__traced_save_5221
Υ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_variance!depthwise_conv2d/depthwise_kernelbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_variance#depthwise_conv2d_1/depthwise_kernelbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceconv2d_2/kernelbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_variance#depthwise_conv2d_2/depthwise_kernelbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceconv2d_3/kernelbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_variance#depthwise_conv2d_3/depthwise_kernelbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_varianceconv2d_4/kernelbatch_normalization_8/gammabatch_normalization_8/beta!batch_normalization_8/moving_mean%batch_normalization_8/moving_variancedense/kernel
dense/bias*;
Tin4
220*
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
 __inference__traced_restore_5372δΏ
γ

.__inference_depthwise_conv2d_layer_call_fn_832

inputs%
!depthwise_conv2d_depthwise_kernel
identity’StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputs!depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_8282
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? :22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
―
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_2183

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ΊΓ
΅
F__inference_functional_1_layer_call_and_return_conditional_losses_3771

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimΜ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDims―
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpΤ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2DΑ
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpΖ
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ϊ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ι
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6κ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape₯
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rateω
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwiseΙ
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpΞ
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1β
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6·
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpΚ
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2DΙ
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOpΞ
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Χ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6ς
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp‘
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape©
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwiseΙ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpΞ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6·
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΚ
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2DΙ
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpΞ
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Χ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6ς
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp‘
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shape©
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseΙ
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpΞ
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6Έ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΛ
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_3/Conv2DΚ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOpΟ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ά
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_6/Relu6σ
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp‘
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2$
"depthwise_conv2d_3/depthwise/Shape©
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseΚ
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_7/ReadVariableOpΟ
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ι
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_7/Relu6Ή
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpΛ
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_4/Conv2DΚ
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_8/ReadVariableOpΟ
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ά
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_8/Relu6³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesΗ
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	2
global_average_pooling2d/Mean
dropout/IdentityIdentity&global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:	2
dropout/Identity€
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul 
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdda
IdentityIdentitydense/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(::::::::::::::::::::::::::::::::::::::::::::::::S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
ξ
γ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2243

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ζ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
½¨

F__inference_functional_1_layer_call_and_return_conditional_losses_2675

inputs
conv2d_conv2d_kernel1
-batch_normalization_batch_normalization_gamma0
,batch_normalization_batch_normalization_beta7
3batch_normalization_batch_normalization_moving_mean;
7batch_normalization_batch_normalization_moving_variance6
2depthwise_conv2d_depthwise_conv2d_depthwise_kernel5
1batch_normalization_1_batch_normalization_1_gamma4
0batch_normalization_1_batch_normalization_1_beta;
7batch_normalization_1_batch_normalization_1_moving_mean?
;batch_normalization_1_batch_normalization_1_moving_variance
conv2d_1_conv2d_1_kernel5
1batch_normalization_2_batch_normalization_2_gamma4
0batch_normalization_2_batch_normalization_2_beta;
7batch_normalization_2_batch_normalization_2_moving_mean?
;batch_normalization_2_batch_normalization_2_moving_variance:
6depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel5
1batch_normalization_3_batch_normalization_3_gamma4
0batch_normalization_3_batch_normalization_3_beta;
7batch_normalization_3_batch_normalization_3_moving_mean?
;batch_normalization_3_batch_normalization_3_moving_variance
conv2d_2_conv2d_2_kernel5
1batch_normalization_4_batch_normalization_4_gamma4
0batch_normalization_4_batch_normalization_4_beta;
7batch_normalization_4_batch_normalization_4_moving_mean?
;batch_normalization_4_batch_normalization_4_moving_variance:
6depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel5
1batch_normalization_5_batch_normalization_5_gamma4
0batch_normalization_5_batch_normalization_5_beta;
7batch_normalization_5_batch_normalization_5_moving_mean?
;batch_normalization_5_batch_normalization_5_moving_variance
conv2d_3_conv2d_3_kernel5
1batch_normalization_6_batch_normalization_6_gamma4
0batch_normalization_6_batch_normalization_6_beta;
7batch_normalization_6_batch_normalization_6_moving_mean?
;batch_normalization_6_batch_normalization_6_moving_variance:
6depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel5
1batch_normalization_7_batch_normalization_7_gamma4
0batch_normalization_7_batch_normalization_7_beta;
7batch_normalization_7_batch_normalization_7_moving_mean?
;batch_normalization_7_batch_normalization_7_moving_variance
conv2d_4_conv2d_4_kernel5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
dense_dense_kernel
dense_dense_bias
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’-batch_normalization_8/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’dense/StatefulPartitionedCall’(depthwise_conv2d/StatefulPartitionedCall’*depthwise_conv2d_1/StatefulPartitionedCall’*depthwise_conv2d_2/StatefulPartitionedCall’*depthwise_conv2d_3/StatefulPartitionedCall’dropout/StatefulPartitionedCallϋ
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_16752(
&tf_op_layer_ExpandDims/PartitionedCall¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0conv2d_conv2d_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_16902 
conv2d/StatefulPartitionedCallώ
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17182-
+batch_normalization/StatefulPartitionedCallφ
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_17652
re_lu/PartitionedCallΠ
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:02depthwise_conv2d_depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_8282*
(depthwise_conv2d/StatefulPartitionedCall
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17982/
-batch_normalization_1/StatefulPartitionedCallώ
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_18452
re_lu_1/PartitionedCall‘
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_1_conv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18602"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_18872/
-batch_normalization_2/StatefulPartitionedCallώ
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_19342
re_lu_2/PartitionedCallέ
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:06depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_10382,
*depthwise_conv2d_1/StatefulPartitionedCall 
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19672/
-batch_normalization_3/StatefulPartitionedCallώ
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_20142
re_lu_3/PartitionedCall‘
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_2_conv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_20292"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20562/
-batch_normalization_4/StatefulPartitionedCallώ
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_21032
re_lu_4/PartitionedCallέ
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:06depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_12482,
*depthwise_conv2d_2/StatefulPartitionedCall 
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:01batch_normalization_5_batch_normalization_5_gamma0batch_normalization_5_batch_normalization_5_beta7batch_normalization_5_batch_normalization_5_moving_mean;batch_normalization_5_batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21362/
-batch_normalization_5/StatefulPartitionedCallώ
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_21832
re_lu_5/PartitionedCall’
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_3_conv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_21982"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_6_batch_normalization_6_gamma0batch_normalization_6_batch_normalization_6_beta7batch_normalization_6_batch_normalization_6_moving_mean;batch_normalization_6_batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22252/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_22722
re_lu_6/PartitionedCallή
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:06depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_14582,
*depthwise_conv2d_3/StatefulPartitionedCall‘
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:01batch_normalization_7_batch_normalization_7_gamma0batch_normalization_7_batch_normalization_7_beta7batch_normalization_7_batch_normalization_7_moving_mean;batch_normalization_7_batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23052/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_23522
re_lu_7/PartitionedCall’
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv2d_4_conv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_23672"
 conv2d_4/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_23942/
-batch_normalization_8/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_24412
re_lu_8/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_16622*
(global_average_pooling2d/PartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24672!
dropout/StatefulPartitionedCall¦
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_24952
dense/StatefulPartitionedCallΐ
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall ^dropout/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(:::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
χ

O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4836

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
χ

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1402

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
	
¨
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_816

inputs>
:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel
identity·
depthwise/ReadVariableOpReadVariableOp:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΞ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? ::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4302

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
³
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_4759

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1816

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ξ
γ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4736

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ζ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Θ
u
'__inference_conv2d_4_layer_call_fn_4895

inputs
conv2d_4_kernel
identity’StatefulPartitionedCallν
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_23672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0**
_input_shapes
::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs

Q
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3886

inputs
identityΝ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_16752
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
Ο

B__inference_conv2d_4_layer_call_and_return_conditional_losses_2367

inputs)
%conv2d_readvariableop_conv2d_4_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0**
_input_shapes
:::O K
'
_output_shapes
:
 
_user_specified_nameinputs
½
Ω
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3935

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ξ
γ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4800

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ζ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
α

4__inference_batch_normalization_5_layer_call_fn_4569

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identity’StatefulPartitionedCallρ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
³
]
A__inference_re_lu_6_layer_call_and_return_conditional_losses_2272

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Μ

B__inference_conv2d_3_layer_call_and_return_conditional_losses_2198

inputs)
%conv2d_readvariableop_conv2d_3_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*)
_input_shapes
:@::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
?

O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_1798

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
―
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4261

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ε
ύ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3971

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΓ
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ι

B__inference_conv2d_2_layer_call_and_return_conditional_losses_4391

inputs)
%conv2d_readvariableop_conv2d_2_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Έ
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1662

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
€
_
A__inference_dropout_layer_call_and_return_conditional_losses_2472

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
	
­
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1236

inputs@
<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel
identityΉ
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΝ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
­
[
?__inference_re_lu_layer_call_and_return_conditional_losses_1765

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs

B
&__inference_re_lu_7_layer_call_fn_4882

inputs
identityΏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_23522
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Λ	

4__inference_batch_normalization_3_layer_call_fn_4311

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11002
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
€
_
A__inference_dropout_layer_call_and_return_conditional_losses_5030

inputs

identity_1R
IdentityIdentityinputs*
T0*
_output_shapes
:	2

Identitya

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
:	2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
γ

4__inference_batch_normalization_6_layer_call_fn_4745

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identity’StatefulPartitionedCallπ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22252
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ρ	

4__inference_batch_normalization_7_layer_call_fn_4872

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_15472
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
	
­
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1446

inputs@
<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel
identityΊ
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΞ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,???????????????????????????::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs

S
7__inference_global_average_pooling2d_layer_call_fn_1665

inputs
identityΩ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_16622
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs

B
&__inference_re_lu_6_layer_call_fn_4764

inputs
identityΏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_22722
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
?

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2136

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Π
Ω
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3989

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4433

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs

B
&__inference_re_lu_5_layer_call_fn_4633

inputs
identityΎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_21832
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
	
­
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1248

inputs@
<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel
identityΉ
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΝ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ι

B__inference_conv2d_2_layer_call_and_return_conditional_losses_2029

inputs)
%conv2d_readvariableop_conv2d_2_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
χ

O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4967

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ό
Ψ
L__inference_batch_normalization_layer_call_and_return_conditional_losses_799

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ϋ
γ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1547

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ν	

4__inference_batch_normalization_5_layer_call_fn_4623

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_13372
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs

B
&__inference_re_lu_8_layer_call_fn_5013

inputs
identityΏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_24412
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs

B
&__inference_re_lu_2_layer_call_fn_4266

inputs
identityΎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_19342
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2074

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
 
²
+__inference_functional_1_layer_call_fn_3823

inputs
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identity’StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*?
_read_only_resource_inputs!
 !$%&)*+./*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_26752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*θ
_input_shapesΦ
Σ:?????????1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
Μ	

4__inference_batch_normalization_1_layer_call_fn_4125

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9172
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Π
Ω
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1736

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs


O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_2225

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Τ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ρ
ύ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3917

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΓ
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
―
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_2103

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ό
q
%__inference_conv2d_layer_call_fn_3899

inputs
conv2d_kernel
identity’StatefulPartitionedCallθ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_16902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:1(:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
Έ	
φ
2__inference_batch_normalization_layer_call_fn_3953

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_7992
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4107

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
κ

N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_982

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Κ	

4__inference_batch_normalization_2_layer_call_fn_4247

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9822
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ι

B__inference_conv2d_1_layer_call_and_return_conditional_losses_4142

inputs)
%conv2d_readvariableop_conv2d_1_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ό

$__inference_dense_layer_call_fn_5057

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

:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_24952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*&
_input_shapes
:	::22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4587

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4089

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4415

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1310

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
χ

O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4664

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ε’
#
F__inference_functional_1_layer_call_and_return_conditional_losses_3586

inputs.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity’"batch_normalization/AssignNewValue’$batch_normalization/AssignNewValue_1’$batch_normalization_1/AssignNewValue’&batch_normalization_1/AssignNewValue_1’$batch_normalization_2/AssignNewValue’&batch_normalization_2/AssignNewValue_1’$batch_normalization_3/AssignNewValue’&batch_normalization_3/AssignNewValue_1’$batch_normalization_4/AssignNewValue’&batch_normalization_4/AssignNewValue_1’$batch_normalization_5/AssignNewValue’&batch_normalization_5/AssignNewValue_1’$batch_normalization_6/AssignNewValue’&batch_normalization_6/AssignNewValue_1’$batch_normalization_7/AssignNewValue’&batch_normalization_7/AssignNewValue_1’$batch_normalization_8/AssignNewValue’&batch_normalization_8/AssignNewValue_1
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimΜ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinputs.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDims―
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpΤ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2DΑ
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpΖ
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ϊ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Χ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2&
$batch_normalization/FusedBatchNormV3₯
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*f
_class\
ZXloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue»
$batch_normalization/AssignNewValue_1AssignVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*l
_classb
`^loc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6κ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape₯
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rateω
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwiseΙ
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpΞ
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1π
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_1/FusedBatchNormV3΅
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueΛ
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6·
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpΚ
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2DΙ
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOpΞ
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ε
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_2/FusedBatchNormV3΅
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueΛ
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6ς
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp‘
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape©
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwiseΙ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpΞ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_3/FusedBatchNormV3΅
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueΛ
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6·
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΚ
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2DΙ
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpΞ
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ε
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_4/FusedBatchNormV3΅
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValueΛ
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6ς
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp‘
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shape©
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseΙ
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpΞ
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_5/FusedBatchNormV3΅
$batch_normalization_5/AssignNewValueAssignVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValueΛ
&batch_normalization_5/AssignNewValue_1AssignVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6Έ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΛ
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_3/Conv2DΚ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOpΟ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1κ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_6/FusedBatchNormV3΅
$batch_normalization_6/AssignNewValueAssignVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValueΛ
&batch_normalization_6/AssignNewValue_1AssignVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_6/Relu6σ
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp‘
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2$
"depthwise_conv2d_3/depthwise/Shape©
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseΚ
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_7/ReadVariableOpΟ
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1χ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_7/FusedBatchNormV3΅
$batch_normalization_7/AssignNewValueAssignVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValueΛ
&batch_normalization_7/AssignNewValue_1AssignVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_7/Relu6Ή
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpΛ
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_4/Conv2DΚ
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_8/ReadVariableOpΟ
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1κ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_8/FusedBatchNormV3΅
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValueΛ
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_8/Relu6³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesΗ
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	2
global_average_pooling2d/Means
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const£
dropout/dropout/MulMul&global_average_pooling2d/Mean:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/dropout/ShapeΔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2 
dropout/dropout/GreaterEqual/yΦ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1€
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul 
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdd­
IdentityIdentitydense/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(:::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
α

4__inference_batch_normalization_2_layer_call_fn_4202

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identity’StatefulPartitionedCallρ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ρ

1__inference_depthwise_conv2d_3_layer_call_fn_1462

inputs'
#depthwise_conv2d_3_depthwise_kernel
identity’StatefulPartitionedCall¦
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_14582
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,???????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Δ
u
'__inference_conv2d_2_layer_call_fn_4397

inputs
conv2d_2_kernel
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_20292
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ϋ
γ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4682

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1009

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ι

B__inference_conv2d_1_layer_call_and_return_conditional_losses_1860

inputs)
%conv2d_readvariableop_conv2d_1_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
: ::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1905

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
―
]
A__inference_re_lu_2_layer_call_and_return_conditional_losses_1934

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
 
³
+__inference_functional_1_layer_call_fn_3342
input_1
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identity’StatefulPartitionedCall΄
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*?
_read_only_resource_inputs!
 !$%&)*+./*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_26752
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*θ
_input_shapesΦ
Σ:?????????1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1


O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2394

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Τ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ν	

4__inference_batch_normalization_2_layer_call_fn_4256

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_10092
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Λ
φ
2__inference_batch_normalization_layer_call_fn_3998

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
κ

N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_890

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
§§
ρ
F__inference_functional_1_layer_call_and_return_conditional_losses_2809

inputs
conv2d_conv2d_kernel1
-batch_normalization_batch_normalization_gamma0
,batch_normalization_batch_normalization_beta7
3batch_normalization_batch_normalization_moving_mean;
7batch_normalization_batch_normalization_moving_variance6
2depthwise_conv2d_depthwise_conv2d_depthwise_kernel5
1batch_normalization_1_batch_normalization_1_gamma4
0batch_normalization_1_batch_normalization_1_beta;
7batch_normalization_1_batch_normalization_1_moving_mean?
;batch_normalization_1_batch_normalization_1_moving_variance
conv2d_1_conv2d_1_kernel5
1batch_normalization_2_batch_normalization_2_gamma4
0batch_normalization_2_batch_normalization_2_beta;
7batch_normalization_2_batch_normalization_2_moving_mean?
;batch_normalization_2_batch_normalization_2_moving_variance:
6depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel5
1batch_normalization_3_batch_normalization_3_gamma4
0batch_normalization_3_batch_normalization_3_beta;
7batch_normalization_3_batch_normalization_3_moving_mean?
;batch_normalization_3_batch_normalization_3_moving_variance
conv2d_2_conv2d_2_kernel5
1batch_normalization_4_batch_normalization_4_gamma4
0batch_normalization_4_batch_normalization_4_beta;
7batch_normalization_4_batch_normalization_4_moving_mean?
;batch_normalization_4_batch_normalization_4_moving_variance:
6depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel5
1batch_normalization_5_batch_normalization_5_gamma4
0batch_normalization_5_batch_normalization_5_beta;
7batch_normalization_5_batch_normalization_5_moving_mean?
;batch_normalization_5_batch_normalization_5_moving_variance
conv2d_3_conv2d_3_kernel5
1batch_normalization_6_batch_normalization_6_gamma4
0batch_normalization_6_batch_normalization_6_beta;
7batch_normalization_6_batch_normalization_6_moving_mean?
;batch_normalization_6_batch_normalization_6_moving_variance:
6depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel5
1batch_normalization_7_batch_normalization_7_gamma4
0batch_normalization_7_batch_normalization_7_beta;
7batch_normalization_7_batch_normalization_7_moving_mean?
;batch_normalization_7_batch_normalization_7_moving_variance
conv2d_4_conv2d_4_kernel5
1batch_normalization_8_batch_normalization_8_gamma4
0batch_normalization_8_batch_normalization_8_beta;
7batch_normalization_8_batch_normalization_8_moving_mean?
;batch_normalization_8_batch_normalization_8_moving_variance
dense_dense_kernel
dense_dense_bias
identity’+batch_normalization/StatefulPartitionedCall’-batch_normalization_1/StatefulPartitionedCall’-batch_normalization_2/StatefulPartitionedCall’-batch_normalization_3/StatefulPartitionedCall’-batch_normalization_4/StatefulPartitionedCall’-batch_normalization_5/StatefulPartitionedCall’-batch_normalization_6/StatefulPartitionedCall’-batch_normalization_7/StatefulPartitionedCall’-batch_normalization_8/StatefulPartitionedCall’conv2d/StatefulPartitionedCall’ conv2d_1/StatefulPartitionedCall’ conv2d_2/StatefulPartitionedCall’ conv2d_3/StatefulPartitionedCall’ conv2d_4/StatefulPartitionedCall’dense/StatefulPartitionedCall’(depthwise_conv2d/StatefulPartitionedCall’*depthwise_conv2d_1/StatefulPartitionedCall’*depthwise_conv2d_2/StatefulPartitionedCall’*depthwise_conv2d_3/StatefulPartitionedCallϋ
&tf_op_layer_ExpandDims/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:1(* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *Y
fTRR
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_16752(
&tf_op_layer_ExpandDims/PartitionedCall¦
conv2d/StatefulPartitionedCallStatefulPartitionedCall/tf_op_layer_ExpandDims/PartitionedCall:output:0conv2d_conv2d_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_16902 
conv2d/StatefulPartitionedCall
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0-batch_normalization_batch_normalization_gamma,batch_normalization_batch_normalization_beta3batch_normalization_batch_normalization_moving_mean7batch_normalization_batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17362-
+batch_normalization/StatefulPartitionedCallφ
re_lu/PartitionedCallPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_17652
re_lu/PartitionedCallΠ
(depthwise_conv2d/StatefulPartitionedCallStatefulPartitionedCallre_lu/PartitionedCall:output:02depthwise_conv2d_depthwise_conv2d_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_8282*
(depthwise_conv2d/StatefulPartitionedCall 
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall1depthwise_conv2d/StatefulPartitionedCall:output:01batch_normalization_1_batch_normalization_1_gamma0batch_normalization_1_batch_normalization_1_beta7batch_normalization_1_batch_normalization_1_moving_mean;batch_normalization_1_batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_18162/
-batch_normalization_1/StatefulPartitionedCallώ
re_lu_1/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_18452
re_lu_1/PartitionedCall‘
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_1/PartitionedCall:output:0conv2d_1_conv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18602"
 conv2d_1/StatefulPartitionedCall
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:01batch_normalization_2_batch_normalization_2_gamma0batch_normalization_2_batch_normalization_2_beta7batch_normalization_2_batch_normalization_2_moving_mean;batch_normalization_2_batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_19052/
-batch_normalization_2/StatefulPartitionedCallώ
re_lu_2/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_2_layer_call_and_return_conditional_losses_19342
re_lu_2/PartitionedCallέ
*depthwise_conv2d_1/StatefulPartitionedCallStatefulPartitionedCall re_lu_2/PartitionedCall:output:06depthwise_conv2d_1_depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_10382,
*depthwise_conv2d_1/StatefulPartitionedCall’
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_1/StatefulPartitionedCall:output:01batch_normalization_3_batch_normalization_3_gamma0batch_normalization_3_batch_normalization_3_beta7batch_normalization_3_batch_normalization_3_moving_mean;batch_normalization_3_batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19852/
-batch_normalization_3/StatefulPartitionedCallώ
re_lu_3/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_20142
re_lu_3/PartitionedCall‘
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_3/PartitionedCall:output:0conv2d_2_conv2d_2_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_20292"
 conv2d_2/StatefulPartitionedCall
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:01batch_normalization_4_batch_normalization_4_gamma0batch_normalization_4_batch_normalization_4_beta7batch_normalization_4_batch_normalization_4_moving_mean;batch_normalization_4_batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20742/
-batch_normalization_4/StatefulPartitionedCallώ
re_lu_4/PartitionedCallPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_21032
re_lu_4/PartitionedCallέ
*depthwise_conv2d_2/StatefulPartitionedCallStatefulPartitionedCall re_lu_4/PartitionedCall:output:06depthwise_conv2d_2_depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_12482,
*depthwise_conv2d_2/StatefulPartitionedCall’
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_2/StatefulPartitionedCall:output:01batch_normalization_5_batch_normalization_5_gamma0batch_normalization_5_batch_normalization_5_beta7batch_normalization_5_batch_normalization_5_moving_mean;batch_normalization_5_batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21542/
-batch_normalization_5/StatefulPartitionedCallώ
re_lu_5/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_5_layer_call_and_return_conditional_losses_21832
re_lu_5/PartitionedCall’
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_5/PartitionedCall:output:0conv2d_3_conv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_21982"
 conv2d_3/StatefulPartitionedCall
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:01batch_normalization_6_batch_normalization_6_gamma0batch_normalization_6_batch_normalization_6_beta7batch_normalization_6_batch_normalization_6_moving_mean;batch_normalization_6_batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22432/
-batch_normalization_6/StatefulPartitionedCall?
re_lu_6/PartitionedCallPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_6_layer_call_and_return_conditional_losses_22722
re_lu_6/PartitionedCallή
*depthwise_conv2d_3/StatefulPartitionedCallStatefulPartitionedCall re_lu_6/PartitionedCall:output:06depthwise_conv2d_3_depthwise_conv2d_3_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_14582,
*depthwise_conv2d_3/StatefulPartitionedCall£
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall3depthwise_conv2d_3/StatefulPartitionedCall:output:01batch_normalization_7_batch_normalization_7_gamma0batch_normalization_7_batch_normalization_7_beta7batch_normalization_7_batch_normalization_7_moving_mean;batch_normalization_7_batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23232/
-batch_normalization_7/StatefulPartitionedCall?
re_lu_7/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_7_layer_call_and_return_conditional_losses_23522
re_lu_7/PartitionedCall’
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall re_lu_7/PartitionedCall:output:0conv2d_4_conv2d_4_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_4_layer_call_and_return_conditional_losses_23672"
 conv2d_4/StatefulPartitionedCall
-batch_normalization_8/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:01batch_normalization_8_batch_normalization_8_gamma0batch_normalization_8_batch_normalization_8_beta7batch_normalization_8_batch_normalization_8_moving_mean;batch_normalization_8_batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24122/
-batch_normalization_8/StatefulPartitionedCall?
re_lu_8/PartitionedCallPartitionedCall6batch_normalization_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_8_layer_call_and_return_conditional_losses_24412
re_lu_8/PartitionedCall
(global_average_pooling2d/PartitionedCallPartitionedCall re_lu_8/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *[
fVRT
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_16622*
(global_average_pooling2d/PartitionedCallς
dropout/PartitionedCallPartitionedCall1global_average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24722
dropout/PartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_dense_kerneldense_dense_bias*
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
GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_24952
dense/StatefulPartitionedCall
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall.^batch_normalization_8/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall^dense/StatefulPartitionedCall)^depthwise_conv2d/StatefulPartitionedCall+^depthwise_conv2d_1/StatefulPartitionedCall+^depthwise_conv2d_2/StatefulPartitionedCall+^depthwise_conv2d_3/StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(:::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2^
-batch_normalization_8/StatefulPartitionedCall-batch_normalization_8/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2T
(depthwise_conv2d/StatefulPartitionedCall(depthwise_conv2d/StatefulPartitionedCall2X
*depthwise_conv2d_1/StatefulPartitionedCall*depthwise_conv2d_1/StatefulPartitionedCall2X
*depthwise_conv2d_2/StatefulPartitionedCall*depthwise_conv2d_2/StatefulPartitionedCall2X
*depthwise_conv2d_3/StatefulPartitionedCall*depthwise_conv2d_3/StatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs


O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4782

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Τ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:
 
_user_specified_nameinputs
ί

4__inference_batch_normalization_4_layer_call_fn_4442

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ν

1__inference_depthwise_conv2d_1_layer_call_fn_1042

inputs'
#depthwise_conv2d_1_depthwise_kernel
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_1_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_10382
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? :22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4338

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ξ
γ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_2412

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ζ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
γ

4__inference_batch_normalization_8_layer_call_fn_4940

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity’StatefulPartitionedCallπ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_23942
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ϋ
γ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1639

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4469

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
f
­
__inference__traced_save_5221
file_prefix,
(savev2_conv2d_kernel_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop@
<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableopB
>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop:
6savev2_batch_normalization_8_gamma_read_readvariableop9
5savev2_batch_normalization_8_beta_read_readvariableop@
<savev2_batch_normalization_8_moving_mean_read_readvariableopD
@savev2_batch_normalization_8_moving_variance_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop
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
value3B1 B+_temp_2495c199091d4a4cade15864c9525800/part2	
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
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesθ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesπ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop<savev2_depthwise_conv2d_depthwise_kernel_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop>savev2_depthwise_conv2d_1_depthwise_kernel_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop>savev2_depthwise_conv2d_2_depthwise_kernel_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop>savev2_depthwise_conv2d_3_depthwise_kernel_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop6savev2_batch_normalization_8_gamma_read_readvariableop5savev2_batch_normalization_8_beta_read_readvariableop<savev2_batch_normalization_8_moving_mean_read_readvariableop@savev2_batch_normalization_8_moving_variance_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *>
dtypes4
2202
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

identity_1Identity_1:output:0*΄
_input_shapes’
: :( : : : : : : : : : :  : : : : : : : : : : @:@:@:@:@:@:@:@:@:@:@:::::::::::::::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:( : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 	

_output_shapes
: : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-)
'
_output_shapes
:@:! 

_output_shapes	
::!!

_output_shapes	
::!"

_output_shapes	
::!#

_output_shapes	
::-$)
'
_output_shapes
::!%

_output_shapes	
::!&

_output_shapes	
::!'

_output_shapes	
::!(

_output_shapes	
::.)*
(
_output_shapes
::!*

_output_shapes	
::!+

_output_shapes	
::!,

_output_shapes	
::!-

_output_shapes	
::%.!

_output_shapes
:	: /

_output_shapes
::0

_output_shapes
: 
?

O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4166

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Π
ό
L__inference_batch_normalization_layer_call_and_return_conditional_losses_772

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΓ
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ο	

4__inference_batch_normalization_7_layer_call_fn_4863

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_15202
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Κ	

4__inference_batch_normalization_1_layer_call_fn_4116

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *W
fRRP
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_8902
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ψ
_
&__inference_dropout_layer_call_fn_5035

inputs
identity’StatefulPartitionedCallΟ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	22
StatefulPartitionedCallStatefulPartitionedCall:G C

_output_shapes
:	
 
_user_specified_nameinputs

@
$__inference_re_lu_layer_call_fn_4017

inputs
identityΌ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_re_lu_layer_call_and_return_conditional_losses_17652
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_2154

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ά	
φ
2__inference_batch_normalization_layer_call_fn_3944

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identity’StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_batch_normalization_layer_call_and_return_conditional_losses_7722
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ε
ύ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_1718

inputs,
(readvariableop_batch_normalization_gamma-
)readvariableop_1_batch_normalization_betaC
?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanI
Efusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp(readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp)readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1Ύ
FusedBatchNormV3/ReadVariableOpReadVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΘ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3­
AssignNewValueAssignVariableOp?fusedbatchnormv3_readvariableop_batch_normalization_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*R
_classH
FDloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΓ
AssignNewValue_1AssignVariableOpEfusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*X
_classN
LJloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
½Γ
Ά
F__inference_functional_1_layer_call_and_return_conditional_losses_3290
input_1.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimΝ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDims―
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpΤ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2DΑ
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpΖ
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ϊ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Ι
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2&
$batch_normalization/FusedBatchNormV3~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6κ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape₯
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rateω
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwiseΙ
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpΞ
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1β
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6·
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpΚ
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2DΙ
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOpΞ
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1Χ
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6ς
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp‘
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape©
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwiseΙ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpΞ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6·
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΚ
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2DΙ
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpΞ
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1Χ
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6ς
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp‘
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shape©
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseΙ
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpΞ
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1δ
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6Έ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΛ
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_3/Conv2DΚ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOpΟ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ά
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_6/Relu6σ
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp‘
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2$
"depthwise_conv2d_3/depthwise/Shape©
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseΚ
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_7/ReadVariableOpΟ
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ι
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_7/Relu6Ή
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpΛ
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_4/Conv2DΚ
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_8/ReadVariableOpΟ
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ά
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2(
&batch_normalization_8/FusedBatchNormV3
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_8/Relu6³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesΗ
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	2
global_average_pooling2d/Mean
dropout/IdentityIdentity&global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:	2
dropout/Identity€
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul 
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdda
IdentityIdentitydense/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(::::::::::::::::::::::::::::::::::::::::::::::::T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
?

O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4035

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Δ
u
'__inference_conv2d_1_layer_call_fn_4148

inputs
conv2d_1_kernel
identity’StatefulPartitionedCallμ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_1_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_18602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
: :22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ν	

4__inference_batch_normalization_3_layer_call_fn_4320

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_11272
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
ί

4__inference_batch_normalization_2_layer_call_fn_4193

inputs
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_18872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?

O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4533

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Λ	

4__inference_batch_normalization_4_layer_call_fn_4496

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_11922
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4284

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4551

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@:::::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ο

B__inference_conv2d_4_layer_call_and_return_conditional_losses_4889

inputs)
%conv2d_readvariableop_conv2d_4_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0**
_input_shapes
:::O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ρ	

4__inference_batch_normalization_6_layer_call_fn_4700

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_14292
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
μ
B
&__inference_dropout_layer_call_fn_5040

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
:	* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_24722
PartitionedCalld
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
―
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_2014

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
	
­
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1458

inputs@
<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel
identityΊ
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΞ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,???????????????????????????*
paddingSAME*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*E
_input_shapes4
2:,???????????????????????????::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1127

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
³
]
A__inference_re_lu_7_layer_call_and_return_conditional_losses_2352

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
­
[
?__inference_re_lu_layer_call_and_return_conditional_losses_4012

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs

B
&__inference_re_lu_4_layer_call_fn_4515

inputs
identityΎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_4_layer_call_and_return_conditional_losses_21032
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs

B
&__inference_re_lu_3_layer_call_fn_4384

inputs
identityΎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_3_layer_call_and_return_conditional_losses_20142
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ν	

4__inference_batch_normalization_4_layer_call_fn_4505

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_12192
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Ο	

4__inference_batch_normalization_8_layer_call_fn_4994

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_16122
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1100

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ψ

`
A__inference_dropout_layer_call_and_return_conditional_losses_2467

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Δ

@__inference_conv2d_layer_call_and_return_conditional_losses_1690

inputs'
#conv2d_readvariableop_conv2d_kernel
identity
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:1(::N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_1337

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
χ

O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_1612

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
―
]
A__inference_re_lu_5_layer_call_and_return_conditional_losses_4628

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
³
]
A__inference_re_lu_7_layer_call_and_return_conditional_losses_4877

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4356

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ν
φ
2__inference_batch_normalization_layer_call_fn_4007

inputs
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance
identity’StatefulPartitionedCallη
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *V
fQRO
M__inference_batch_normalization_layer_call_and_return_conditional_losses_17362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ξ
γ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2323

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ζ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
 
³
+__inference_functional_1_layer_call_fn_3394
input_1
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identity’StatefulPartitionedCallΖ
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_28092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*θ
_input_shapesΦ
Σ:?????????1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
³
]
A__inference_re_lu_8_layer_call_and_return_conditional_losses_2441

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
γ

4__inference_batch_normalization_7_layer_call_fn_4809

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identity’StatefulPartitionedCallπ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23052
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ξ
β
N__inference_batch_normalization_1_layer_call_and_return_conditional_losses_917

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
©
­
?__inference_dense_layer_call_and_return_conditional_losses_2495

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4487

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4053

inputs.
*readvariableop_batch_normalization_1_gamma/
+readvariableop_1_batch_normalization_1_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_1_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4605

inputs.
*readvariableop_batch_normalization_5_gamma/
+readvariableop_1_batch_normalization_5_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_5_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ν

1__inference_depthwise_conv2d_2_layer_call_fn_1252

inputs'
#depthwise_conv2d_2_depthwise_kernel
identity’StatefulPartitionedCall₯
StatefulPartitionedCallStatefulPartitionedCallinputs#depthwise_conv2d_2_depthwise_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_12482
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+???????????????????????????@:22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ί

4__inference_batch_normalization_3_layer_call_fn_4365

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ί

4__inference_batch_normalization_5_layer_call_fn_4560

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_21362
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Ψ
ͺ
"__inference_signature_wrapper_2913
input_1
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__wrapped_model_7142
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:K G
"
_output_shapes
:1(
!
_user_specified_name	input_1
―
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4130

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ρ	

4__inference_batch_normalization_8_layer_call_fn_5003

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_16392
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
―
]
A__inference_re_lu_1_layer_call_and_return_conditional_losses_1845

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
³
]
A__inference_re_lu_8_layer_call_and_return_conditional_losses_5008

inputs
identityQ
Relu6Relu6inputs*
T0*'
_output_shapes
:2
Relu6g
IdentityIdentityRelu6:activations:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*&
_input_shapes
::O K
'
_output_shapes
:
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4220

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ψ

`
A__inference_dropout_layer_call_and_return_conditional_losses_5025

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/Constk
dropout/MulMulinputsdropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/Mulo
dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/Shape¬
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2
dropout/GreaterEqual/yΆ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/GreaterEqualw
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/Castr
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/Mul_1]
IdentityIdentitydropout/Mul_1:z:0*
T0*
_output_shapes
:	2

Identity"
identityIdentity:output:0*
_input_shapes
:	:G C

_output_shapes
:	
 
_user_specified_nameinputs
Έ
n
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1653

inputs
identity
Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2
Mean/reduction_indicesx
MeanMeaninputsMean/reduction_indices:output:0*
T0*0
_output_shapes
:??????????????????2
Meanj
IdentityIdentityMean:output:0*
T0*0
_output_shapes
:??????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
	
­
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1038

inputs@
<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel
identityΉ
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΞ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? ::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_1675

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
Ϋ
γ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4854

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ζ
u
'__inference_conv2d_3_layer_call_fn_4646

inputs
conv2d_3_kernel
identity’StatefulPartitionedCallν
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_kernel*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_3_layer_call_and_return_conditional_losses_21982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*)
_input_shapes
:@:22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
 
²
+__inference_functional_1_layer_call_fn_3875

inputs
conv2d_kernel
batch_normalization_gamma
batch_normalization_beta#
batch_normalization_moving_mean'
#batch_normalization_moving_variance%
!depthwise_conv2d_depthwise_kernel
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
conv2d_1_kernel
batch_normalization_2_gamma
batch_normalization_2_beta%
!batch_normalization_2_moving_mean)
%batch_normalization_2_moving_variance'
#depthwise_conv2d_1_depthwise_kernel
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
conv2d_2_kernel
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance'
#depthwise_conv2d_2_depthwise_kernel
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
conv2d_3_kernel
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance'
#depthwise_conv2d_3_depthwise_kernel
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
conv2d_4_kernel
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
dense_kernel

dense_bias
identity’StatefulPartitionedCallΕ
StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_kernelbatch_normalization_gammabatch_normalization_betabatch_normalization_moving_mean#batch_normalization_moving_variance!depthwise_conv2d_depthwise_kernelbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_varianceconv2d_1_kernelbatch_normalization_2_gammabatch_normalization_2_beta!batch_normalization_2_moving_mean%batch_normalization_2_moving_variance#depthwise_conv2d_1_depthwise_kernelbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_varianceconv2d_2_kernelbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance#depthwise_conv2d_2_depthwise_kernelbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_varianceconv2d_3_kernelbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance#depthwise_conv2d_3_depthwise_kernelbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_varianceconv2d_4_kernelbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variancedense_kernel
dense_bias*;
Tin4
220*
Tout
2*
_collective_manager_ids
 *
_output_shapes

:*Q
_read_only_resource_inputs3
1/	
 !"#$%&'()*+,-./*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_functional_1_layer_call_and_return_conditional_losses_28092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*θ
_input_shapesΦ
Σ:?????????1(:::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????1(
 
_user_specified_nameinputs
	
­
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1026

inputs@
<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel
identityΉ
depthwise/ReadVariableOpReadVariableOp<depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΞ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? ::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs

B
&__inference_re_lu_1_layer_call_fn_4135

inputs
identityΎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_re_lu_1_layer_call_and_return_conditional_losses_18452
PartitionedCallk
IdentityIdentityPartitionedCall:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Λ	

4__inference_batch_normalization_5_layer_call_fn_4614

inputs
batch_normalization_5_gamma
batch_normalization_5_beta%
!batch_normalization_5_moving_mean)
%batch_normalization_5_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_5_gammabatch_normalization_5_beta!batch_normalization_5_moving_mean%batch_normalization_5_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_13102
StatefulPartitionedCall¨
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ξ
γ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4931

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ζ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3h
IdentityIdentityFusedBatchNormV3:y:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#::::::O K
'
_output_shapes
:
 
_user_specified_nameinputs
α

4__inference_batch_normalization_4_layer_call_fn_4451

inputs
batch_normalization_4_gamma
batch_normalization_4_beta%
!batch_normalization_4_moving_mean)
%batch_normalization_4_moving_variance
identity’StatefulPartitionedCallρ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_4_gammabatch_normalization_4_beta!batch_normalization_4_moving_mean%batch_normalization_4_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
:@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_20742
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
:@
 
_user_specified_nameinputs


O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_2305

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Τ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:
 
_user_specified_nameinputs
Ϋ
γ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_1429

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
Ο	

4__inference_batch_normalization_6_layer_call_fn_4691

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,???????????????????????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_14022
StatefulPartitionedCall©
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
χ

O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_1520

inputs.
*readvariableop_batch_normalization_7_gamma/
+readvariableop_1_batch_normalization_7_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_7_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1§
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1985

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
α

4__inference_batch_normalization_3_layer_call_fn_4374

inputs
batch_normalization_3_gamma
batch_normalization_3_beta%
!batch_normalization_3_moving_mean)
%batch_normalization_3_moving_variance
identity’StatefulPartitionedCallρ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_3_gammabatch_normalization_3_beta!batch_normalization_3_moving_mean%batch_normalization_3_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_19852
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs


O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4913

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Τ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:
 
_user_specified_nameinputs


O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4718

inputs.
*readvariableop_batch_normalization_6_gamma/
+readvariableop_1_batch_normalization_6_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Τ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_6_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:O K
'
_output_shapes
:
 
_user_specified_nameinputs
η
ρ!
__inference__wrapped_model_714
input_1;
7functional_1_conv2d_conv2d_readvariableop_conv2d_kernelM
Ifunctional_1_batch_normalization_readvariableop_batch_normalization_gammaN
Jfunctional_1_batch_normalization_readvariableop_1_batch_normalization_betad
`functional_1_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_meanj
ffunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance\
Xfunctional_1_depthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelQ
Mfunctional_1_batch_normalization_1_readvariableop_batch_normalization_1_gammaR
Nfunctional_1_batch_normalization_1_readvariableop_1_batch_normalization_1_betah
dfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meann
jfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance?
;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernelQ
Mfunctional_1_batch_normalization_2_readvariableop_batch_normalization_2_gammaR
Nfunctional_1_batch_normalization_2_readvariableop_1_batch_normalization_2_betah
dfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meann
jfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance`
\functional_1_depthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelQ
Mfunctional_1_batch_normalization_3_readvariableop_batch_normalization_3_gammaR
Nfunctional_1_batch_normalization_3_readvariableop_1_batch_normalization_3_betah
dfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meann
jfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance?
;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernelQ
Mfunctional_1_batch_normalization_4_readvariableop_batch_normalization_4_gammaR
Nfunctional_1_batch_normalization_4_readvariableop_1_batch_normalization_4_betah
dfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meann
jfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance`
\functional_1_depthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelQ
Mfunctional_1_batch_normalization_5_readvariableop_batch_normalization_5_gammaR
Nfunctional_1_batch_normalization_5_readvariableop_1_batch_normalization_5_betah
dfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meann
jfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance?
;functional_1_conv2d_3_conv2d_readvariableop_conv2d_3_kernelQ
Mfunctional_1_batch_normalization_6_readvariableop_batch_normalization_6_gammaR
Nfunctional_1_batch_normalization_6_readvariableop_1_batch_normalization_6_betah
dfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meann
jfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance`
\functional_1_depthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelQ
Mfunctional_1_batch_normalization_7_readvariableop_batch_normalization_7_gammaR
Nfunctional_1_batch_normalization_7_readvariableop_1_batch_normalization_7_betah
dfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meann
jfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance?
;functional_1_conv2d_4_conv2d_readvariableop_conv2d_4_kernelQ
Mfunctional_1_batch_normalization_8_readvariableop_batch_normalization_8_gammaR
Nfunctional_1_batch_normalization_8_readvariableop_1_batch_normalization_8_betah
dfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meann
jfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance9
5functional_1_dense_matmul_readvariableop_dense_kernel8
4functional_1_dense_biasadd_readvariableop_dense_bias
identityͺ
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :24
2functional_1/tf_op_layer_ExpandDims/ExpandDims/dimτ
.functional_1/tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1;functional_1/tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(20
.functional_1/tf_op_layer_ExpandDims/ExpandDimsΦ
)functional_1/conv2d/Conv2D/ReadVariableOpReadVariableOp7functional_1_conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02+
)functional_1/conv2d/Conv2D/ReadVariableOp
functional_1/conv2d/Conv2DConv2D7functional_1/tf_op_layer_ExpandDims/ExpandDims:output:01functional_1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
functional_1/conv2d/Conv2Dθ
/functional_1/batch_normalization/ReadVariableOpReadVariableOpIfunctional_1_batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype021
/functional_1/batch_normalization/ReadVariableOpν
1functional_1/batch_normalization/ReadVariableOp_1ReadVariableOpJfunctional_1_batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype023
1functional_1/batch_normalization/ReadVariableOp_1‘
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp`functional_1_batch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype02B
@functional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp«
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpffunctional_1_batch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1€
1functional_1/batch_normalization/FusedBatchNormV3FusedBatchNormV3#functional_1/conv2d/Conv2D:output:07functional_1/batch_normalization/ReadVariableOp:value:09functional_1/batch_normalization/ReadVariableOp_1:value:0Hfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0Jfunctional_1/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 23
1functional_1/batch_normalization/FusedBatchNormV3₯
functional_1/re_lu/Relu6Relu65functional_1/batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu/Relu6
6functional_1/depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpXfunctional_1_depthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype028
6functional_1/depthwise_conv2d/depthwise/ReadVariableOp·
-functional_1/depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2/
-functional_1/depthwise_conv2d/depthwise/ShapeΏ
5functional_1/depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      27
5functional_1/depthwise_conv2d/depthwise/dilation_rate­
'functional_1/depthwise_conv2d/depthwiseDepthwiseConv2dNative&functional_1/re_lu/Relu6:activations:0>functional_1/depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2)
'functional_1/depthwise_conv2d/depthwiseπ
1functional_1/batch_normalization_1/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_1/ReadVariableOpυ
3functional_1/batch_normalization_1/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_1/ReadVariableOp_1©
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp³
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1½
3functional_1/batch_normalization_1/FusedBatchNormV3FusedBatchNormV30functional_1/depthwise_conv2d/depthwise:output:09functional_1/batch_normalization_1/ReadVariableOp:value:0;functional_1/batch_normalization_1/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_1/FusedBatchNormV3«
functional_1/re_lu_1/Relu6Relu67functional_1/batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu_1/Relu6ή
+functional_1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02-
+functional_1/conv2d_1/Conv2D/ReadVariableOpώ
functional_1/conv2d_1/Conv2DConv2D(functional_1/re_lu_1/Relu6:activations:03functional_1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
functional_1/conv2d_1/Conv2Dπ
1functional_1/batch_normalization_2/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_2/ReadVariableOpυ
3functional_1/batch_normalization_2/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_2/ReadVariableOp_1©
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp³
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1²
3functional_1/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_1/Conv2D:output:09functional_1/batch_normalization_2/ReadVariableOp:value:0;functional_1/batch_normalization_2/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_2/FusedBatchNormV3«
functional_1/re_lu_2/Relu6Relu67functional_1/batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu_2/Relu6
8functional_1/depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02:
8functional_1/depthwise_conv2d_1/depthwise/ReadVariableOp»
/functional_1/depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             21
/functional_1/depthwise_conv2d_1/depthwise/ShapeΓ
7functional_1/depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_1/depthwise/dilation_rate΅
)functional_1/depthwise_conv2d_1/depthwiseDepthwiseConv2dNative(functional_1/re_lu_2/Relu6:activations:0@functional_1/depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2+
)functional_1/depthwise_conv2d_1/depthwiseπ
1functional_1/batch_normalization_3/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype023
1functional_1/batch_normalization_3/ReadVariableOpυ
3functional_1/batch_normalization_3/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype025
3functional_1/batch_normalization_3/ReadVariableOp_1©
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02D
Bfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp³
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02F
Dfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1Ώ
3functional_1/batch_normalization_3/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_1/depthwise:output:09functional_1/batch_normalization_3/ReadVariableOp:value:0;functional_1/batch_normalization_3/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_3/FusedBatchNormV3«
functional_1/re_lu_3/Relu6Relu67functional_1/batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
functional_1/re_lu_3/Relu6ή
+functional_1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02-
+functional_1/conv2d_2/Conv2D/ReadVariableOpώ
functional_1/conv2d_2/Conv2DConv2D(functional_1/re_lu_3/Relu6:activations:03functional_1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
functional_1/conv2d_2/Conv2Dπ
1functional_1/batch_normalization_4/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_4/ReadVariableOpυ
3functional_1/batch_normalization_4/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_4/ReadVariableOp_1©
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp³
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1²
3functional_1/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_2/Conv2D:output:09functional_1/batch_normalization_4/ReadVariableOp:value:0;functional_1/batch_normalization_4/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_4/FusedBatchNormV3«
functional_1/re_lu_4/Relu6Relu67functional_1/batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
functional_1/re_lu_4/Relu6
8functional_1/depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02:
8functional_1/depthwise_conv2d_2/depthwise/ReadVariableOp»
/functional_1/depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      21
/functional_1/depthwise_conv2d_2/depthwise/ShapeΓ
7functional_1/depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_2/depthwise/dilation_rate΄
)functional_1/depthwise_conv2d_2/depthwiseDepthwiseConv2dNative(functional_1/re_lu_4/Relu6:activations:0@functional_1/depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_2/depthwiseπ
1functional_1/batch_normalization_5/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype023
1functional_1/batch_normalization_5/ReadVariableOpυ
3functional_1/batch_normalization_5/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype025
3functional_1/batch_normalization_5/ReadVariableOp_1©
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype02D
Bfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp³
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype02F
Dfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1Ώ
3functional_1/batch_normalization_5/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_2/depthwise:output:09functional_1/batch_normalization_5/ReadVariableOp:value:0;functional_1/batch_normalization_5/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_5/FusedBatchNormV3«
functional_1/re_lu_5/Relu6Relu67functional_1/batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
functional_1/re_lu_5/Relu6ί
+functional_1/conv2d_3/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02-
+functional_1/conv2d_3/Conv2D/ReadVariableOp?
functional_1/conv2d_3/Conv2DConv2D(functional_1/re_lu_5/Relu6:activations:03functional_1/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
functional_1/conv2d_3/Conv2Dρ
1functional_1/batch_normalization_6/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_6/ReadVariableOpφ
3functional_1/batch_normalization_6/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype025
3functional_1/batch_normalization_6/ReadVariableOp_1ͺ
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype02D
Bfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp΄
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype02F
Dfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1·
3functional_1/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_3/Conv2D:output:09functional_1/batch_normalization_6/ReadVariableOp:value:0;functional_1/batch_normalization_6/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_6/FusedBatchNormV3¬
functional_1/re_lu_6/Relu6Relu67functional_1/batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
functional_1/re_lu_6/Relu6
8functional_1/depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOp\functional_1_depthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02:
8functional_1/depthwise_conv2d_3/depthwise/ReadVariableOp»
/functional_1/depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            21
/functional_1/depthwise_conv2d_3/depthwise/ShapeΓ
7functional_1/depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_1/depthwise_conv2d_3/depthwise/dilation_rate΅
)functional_1/depthwise_conv2d_3/depthwiseDepthwiseConv2dNative(functional_1/re_lu_6/Relu6:activations:0@functional_1/depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2+
)functional_1/depthwise_conv2d_3/depthwiseρ
1functional_1/batch_normalization_7/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_7/ReadVariableOpφ
3functional_1/batch_normalization_7/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype025
3functional_1/batch_normalization_7/ReadVariableOp_1ͺ
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype02D
Bfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp΄
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype02F
Dfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1Δ
3functional_1/batch_normalization_7/FusedBatchNormV3FusedBatchNormV32functional_1/depthwise_conv2d_3/depthwise:output:09functional_1/batch_normalization_7/ReadVariableOp:value:0;functional_1/batch_normalization_7/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_7/FusedBatchNormV3¬
functional_1/re_lu_7/Relu6Relu67functional_1/batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
functional_1/re_lu_7/Relu6ΰ
+functional_1/conv2d_4/Conv2D/ReadVariableOpReadVariableOp;functional_1_conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02-
+functional_1/conv2d_4/Conv2D/ReadVariableOp?
functional_1/conv2d_4/Conv2DConv2D(functional_1/re_lu_7/Relu6:activations:03functional_1/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
functional_1/conv2d_4/Conv2Dρ
1functional_1/batch_normalization_8/ReadVariableOpReadVariableOpMfunctional_1_batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype023
1functional_1/batch_normalization_8/ReadVariableOpφ
3functional_1/batch_normalization_8/ReadVariableOp_1ReadVariableOpNfunctional_1_batch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype025
3functional_1/batch_normalization_8/ReadVariableOp_1ͺ
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpdfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02D
Bfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp΄
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpjfunctional_1_batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02F
Dfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1·
3functional_1/batch_normalization_8/FusedBatchNormV3FusedBatchNormV3%functional_1/conv2d_4/Conv2D:output:09functional_1/batch_normalization_8/ReadVariableOp:value:0;functional_1/batch_normalization_8/ReadVariableOp_1:value:0Jfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0Lfunctional_1/batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
is_training( 25
3functional_1/batch_normalization_8/FusedBatchNormV3¬
functional_1/re_lu_8/Relu6Relu67functional_1/batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
functional_1/re_lu_8/Relu6Ν
<functional_1/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      2>
<functional_1/global_average_pooling2d/Mean/reduction_indicesϋ
*functional_1/global_average_pooling2d/MeanMean(functional_1/re_lu_8/Relu6:activations:0Efunctional_1/global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	2,
*functional_1/global_average_pooling2d/Mean©
functional_1/dropout/IdentityIdentity3functional_1/global_average_pooling2d/Mean:output:0*
T0*
_output_shapes
:	2
functional_1/dropout/IdentityΛ
(functional_1/dense/MatMul/ReadVariableOpReadVariableOp5functional_1_dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02*
(functional_1/dense/MatMul/ReadVariableOpΓ
functional_1/dense/MatMulMatMul&functional_1/dropout/Identity:output:00functional_1/dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense/MatMulΗ
)functional_1/dense/BiasAdd/ReadVariableOpReadVariableOp4functional_1_dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02+
)functional_1/dense/BiasAdd/ReadVariableOpΔ
functional_1/dense/BiasAddBiasAdd#functional_1/dense/MatMul:product:01functional_1/dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
functional_1/dense/BiasAddn
IdentityIdentity#functional_1/dense/BiasAdd:output:0*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(::::::::::::::::::::::::::::::::::::::::::::::::T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
ί

4__inference_batch_normalization_1_layer_call_fn_4062

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_17982
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
θ’
#
F__inference_functional_1_layer_call_and_return_conditional_losses_3105
input_1.
*conv2d_conv2d_readvariableop_conv2d_kernel@
<batch_normalization_readvariableop_batch_normalization_gammaA
=batch_normalization_readvariableop_1_batch_normalization_betaW
Sbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean]
Ybatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_varianceO
Kdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernelD
@batch_normalization_1_readvariableop_batch_normalization_1_gammaE
Abatch_normalization_1_readvariableop_1_batch_normalization_1_beta[
Wbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_meana
]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance2
.conv2d_1_conv2d_readvariableop_conv2d_1_kernelD
@batch_normalization_2_readvariableop_batch_normalization_2_gammaE
Abatch_normalization_2_readvariableop_1_batch_normalization_2_beta[
Wbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_meana
]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_varianceS
Odepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernelD
@batch_normalization_3_readvariableop_batch_normalization_3_gammaE
Abatch_normalization_3_readvariableop_1_batch_normalization_3_beta[
Wbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_meana
]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance2
.conv2d_2_conv2d_readvariableop_conv2d_2_kernelD
@batch_normalization_4_readvariableop_batch_normalization_4_gammaE
Abatch_normalization_4_readvariableop_1_batch_normalization_4_beta[
Wbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_meana
]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_varianceS
Odepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernelD
@batch_normalization_5_readvariableop_batch_normalization_5_gammaE
Abatch_normalization_5_readvariableop_1_batch_normalization_5_beta[
Wbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_meana
]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance2
.conv2d_3_conv2d_readvariableop_conv2d_3_kernelD
@batch_normalization_6_readvariableop_batch_normalization_6_gammaE
Abatch_normalization_6_readvariableop_1_batch_normalization_6_beta[
Wbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_meana
]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_varianceS
Odepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernelD
@batch_normalization_7_readvariableop_batch_normalization_7_gammaE
Abatch_normalization_7_readvariableop_1_batch_normalization_7_beta[
Wbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_meana
]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance2
.conv2d_4_conv2d_readvariableop_conv2d_4_kernelD
@batch_normalization_8_readvariableop_batch_normalization_8_gammaE
Abatch_normalization_8_readvariableop_1_batch_normalization_8_beta[
Wbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_meana
]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance,
(dense_matmul_readvariableop_dense_kernel+
'dense_biasadd_readvariableop_dense_bias
identity’"batch_normalization/AssignNewValue’$batch_normalization/AssignNewValue_1’$batch_normalization_1/AssignNewValue’&batch_normalization_1/AssignNewValue_1’$batch_normalization_2/AssignNewValue’&batch_normalization_2/AssignNewValue_1’$batch_normalization_3/AssignNewValue’&batch_normalization_3/AssignNewValue_1’$batch_normalization_4/AssignNewValue’&batch_normalization_4/AssignNewValue_1’$batch_normalization_5/AssignNewValue’&batch_normalization_5/AssignNewValue_1’$batch_normalization_6/AssignNewValue’&batch_normalization_6/AssignNewValue_1’$batch_normalization_7/AssignNewValue’&batch_normalization_7/AssignNewValue_1’$batch_normalization_8/AssignNewValue’&batch_normalization_8/AssignNewValue_1
%tf_op_layer_ExpandDims/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2'
%tf_op_layer_ExpandDims/ExpandDims/dimΝ
!tf_op_layer_ExpandDims/ExpandDims
ExpandDimsinput_1.tf_op_layer_ExpandDims/ExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2#
!tf_op_layer_ExpandDims/ExpandDims―
conv2d/Conv2D/ReadVariableOpReadVariableOp*conv2d_conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
conv2d/Conv2D/ReadVariableOpΤ
conv2d/Conv2DConv2D*tf_op_layer_ExpandDims/ExpandDims:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
conv2d/Conv2DΑ
"batch_normalization/ReadVariableOpReadVariableOp<batch_normalization_readvariableop_batch_normalization_gamma*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOpΖ
$batch_normalization/ReadVariableOp_1ReadVariableOp=batch_normalization_readvariableop_1_batch_normalization_beta*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1ϊ
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1Χ
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d/Conv2D:output:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2&
$batch_normalization/FusedBatchNormV3₯
"batch_normalization/AssignNewValueAssignVariableOpSbatch_normalization_fusedbatchnormv3_readvariableop_batch_normalization_moving_mean1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*f
_class\
ZXloc:@batch_normalization/FusedBatchNormV3/ReadVariableOp/batch_normalization/moving_mean*
_output_shapes
 *
dtype02$
"batch_normalization/AssignNewValue»
$batch_normalization/AssignNewValue_1AssignVariableOpYbatch_normalization_fusedbatchnormv3_readvariableop_1_batch_normalization_moving_variance5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*l
_classb
`^loc:@batch_normalization/FusedBatchNormV3/ReadVariableOp_1/batch_normalization/moving_variance*
_output_shapes
 *
dtype02&
$batch_normalization/AssignNewValue_1~
re_lu/Relu6Relu6(batch_normalization/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu/Relu6κ
)depthwise_conv2d/depthwise/ReadVariableOpReadVariableOpKdepthwise_conv2d_depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02+
)depthwise_conv2d/depthwise/ReadVariableOp
 depthwise_conv2d/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2"
 depthwise_conv2d/depthwise/Shape₯
(depthwise_conv2d/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2*
(depthwise_conv2d/depthwise/dilation_rateω
depthwise_conv2d/depthwiseDepthwiseConv2dNativere_lu/Relu6:activations:01depthwise_conv2d/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d/depthwiseΙ
$batch_normalization_1/ReadVariableOpReadVariableOp@batch_normalization_1_readvariableop_batch_normalization_1_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOpΞ
&batch_normalization_1/ReadVariableOp_1ReadVariableOpAbatch_normalization_1_readvariableop_1_batch_normalization_1_beta*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1π
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3#depthwise_conv2d/depthwise:output:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_1/FusedBatchNormV3΅
$batch_normalization_1/AssignNewValueAssignVariableOpWbatch_normalization_1_fusedbatchnormv3_readvariableop_batch_normalization_1_moving_mean3batch_normalization_1/FusedBatchNormV3:batch_mean:06^batch_normalization_1/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp/batch_normalization_1/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_1/AssignNewValueΛ
&batch_normalization_1/AssignNewValue_1AssignVariableOp]batch_normalization_1_fusedbatchnormv3_readvariableop_1_batch_normalization_1_moving_variance7batch_normalization_1/FusedBatchNormV3:batch_variance:08^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_1/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_1/AssignNewValue_1
re_lu_1/Relu6Relu6*batch_normalization_1/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_1/Relu6·
conv2d_1/Conv2D/ReadVariableOpReadVariableOp.conv2d_1_conv2d_readvariableop_conv2d_1_kernel*&
_output_shapes
:  *
dtype02 
conv2d_1/Conv2D/ReadVariableOpΚ
conv2d_1/Conv2DConv2Dre_lu_1/Relu6:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingSAME*
strides
2
conv2d_1/Conv2DΙ
$batch_normalization_2/ReadVariableOpReadVariableOp@batch_normalization_2_readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_2/ReadVariableOpΞ
&batch_normalization_2/ReadVariableOp_1ReadVariableOpAbatch_normalization_2_readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02(
&batch_normalization_2/ReadVariableOp_1
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ε
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3conv2d_1/Conv2D:output:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_2/FusedBatchNormV3΅
$batch_normalization_2/AssignNewValueAssignVariableOpWbatch_normalization_2_fusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean3batch_normalization_2/FusedBatchNormV3:batch_mean:06^batch_normalization_2/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_2/AssignNewValueΛ
&batch_normalization_2/AssignNewValue_1AssignVariableOp]batch_normalization_2_fusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance7batch_normalization_2/FusedBatchNormV3:batch_variance:08^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_2/AssignNewValue_1
re_lu_2/Relu6Relu6*batch_normalization_2/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_2/Relu6ς
+depthwise_conv2d_1/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_1_depthwise_readvariableop_depthwise_conv2d_1_depthwise_kernel*&
_output_shapes
: *
dtype02-
+depthwise_conv2d_1/depthwise/ReadVariableOp‘
"depthwise_conv2d_1/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2$
"depthwise_conv2d_1/depthwise/Shape©
*depthwise_conv2d_1/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_1/depthwise/dilation_rate
depthwise_conv2d_1/depthwiseDepthwiseConv2dNativere_lu_2/Relu6:activations:03depthwise_conv2d_1/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
depthwise_conv2d_1/depthwiseΙ
$batch_normalization_3/ReadVariableOpReadVariableOp@batch_normalization_3_readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02&
$batch_normalization_3/ReadVariableOpΞ
&batch_normalization_3/ReadVariableOp_1ReadVariableOpAbatch_normalization_3_readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02(
&batch_normalization_3/ReadVariableOp_1
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_1/depthwise:output:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_3/FusedBatchNormV3΅
$batch_normalization_3/AssignNewValueAssignVariableOpWbatch_normalization_3_fusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean3batch_normalization_3/FusedBatchNormV3:batch_mean:06^batch_normalization_3/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_3/AssignNewValueΛ
&batch_normalization_3/AssignNewValue_1AssignVariableOp]batch_normalization_3_fusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance7batch_normalization_3/FusedBatchNormV3:batch_variance:08^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_3/AssignNewValue_1
re_lu_3/Relu6Relu6*batch_normalization_3/FusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2
re_lu_3/Relu6·
conv2d_2/Conv2D/ReadVariableOpReadVariableOp.conv2d_2_conv2d_readvariableop_conv2d_2_kernel*&
_output_shapes
: @*
dtype02 
conv2d_2/Conv2D/ReadVariableOpΚ
conv2d_2/Conv2DConv2Dre_lu_3/Relu6:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
conv2d_2/Conv2DΙ
$batch_normalization_4/ReadVariableOpReadVariableOp@batch_normalization_4_readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOpΞ
&batch_normalization_4/ReadVariableOp_1ReadVariableOpAbatch_normalization_4_readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ε
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3conv2d_2/Conv2D:output:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_4/FusedBatchNormV3΅
$batch_normalization_4/AssignNewValueAssignVariableOpWbatch_normalization_4_fusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean3batch_normalization_4/FusedBatchNormV3:batch_mean:06^batch_normalization_4/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_4/AssignNewValueΛ
&batch_normalization_4/AssignNewValue_1AssignVariableOp]batch_normalization_4_fusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance7batch_normalization_4/FusedBatchNormV3:batch_variance:08^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_4/AssignNewValue_1
re_lu_4/Relu6Relu6*batch_normalization_4/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_4/Relu6ς
+depthwise_conv2d_2/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_2_depthwise_readvariableop_depthwise_conv2d_2_depthwise_kernel*&
_output_shapes
:@*
dtype02-
+depthwise_conv2d_2/depthwise/ReadVariableOp‘
"depthwise_conv2d_2/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"      @      2$
"depthwise_conv2d_2/depthwise/Shape©
*depthwise_conv2d_2/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_2/depthwise/dilation_rate
depthwise_conv2d_2/depthwiseDepthwiseConv2dNativere_lu_4/Relu6:activations:03depthwise_conv2d_2/depthwise/ReadVariableOp:value:0*
T0*&
_output_shapes
:@*
paddingSAME*
strides
2
depthwise_conv2d_2/depthwiseΙ
$batch_normalization_5/ReadVariableOpReadVariableOp@batch_normalization_5_readvariableop_batch_normalization_5_gamma*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOpΞ
&batch_normalization_5/ReadVariableOp_1ReadVariableOpAbatch_normalization_5_readvariableop_1_batch_normalization_5_beta*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ς
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_2/depthwise:output:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_5/FusedBatchNormV3΅
$batch_normalization_5/AssignNewValueAssignVariableOpWbatch_normalization_5_fusedbatchnormv3_readvariableop_batch_normalization_5_moving_mean3batch_normalization_5/FusedBatchNormV3:batch_mean:06^batch_normalization_5/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp/batch_normalization_5/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_5/AssignNewValueΛ
&batch_normalization_5/AssignNewValue_1AssignVariableOp]batch_normalization_5_fusedbatchnormv3_readvariableop_1_batch_normalization_5_moving_variance7batch_normalization_5/FusedBatchNormV3:batch_variance:08^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_5/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_5/AssignNewValue_1
re_lu_5/Relu6Relu6*batch_normalization_5/FusedBatchNormV3:y:0*
T0*&
_output_shapes
:@2
re_lu_5/Relu6Έ
conv2d_3/Conv2D/ReadVariableOpReadVariableOp.conv2d_3_conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02 
conv2d_3/Conv2D/ReadVariableOpΛ
conv2d_3/Conv2DConv2Dre_lu_5/Relu6:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_3/Conv2DΚ
$batch_normalization_6/ReadVariableOpReadVariableOp@batch_normalization_6_readvariableop_batch_normalization_6_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_6/ReadVariableOpΟ
&batch_normalization_6/ReadVariableOp_1ReadVariableOpAbatch_normalization_6_readvariableop_1_batch_normalization_6_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_6/ReadVariableOp_1
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1κ
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3conv2d_3/Conv2D:output:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_6/FusedBatchNormV3΅
$batch_normalization_6/AssignNewValueAssignVariableOpWbatch_normalization_6_fusedbatchnormv3_readvariableop_batch_normalization_6_moving_mean3batch_normalization_6/FusedBatchNormV3:batch_mean:06^batch_normalization_6/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp/batch_normalization_6/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_6/AssignNewValueΛ
&batch_normalization_6/AssignNewValue_1AssignVariableOp]batch_normalization_6_fusedbatchnormv3_readvariableop_1_batch_normalization_6_moving_variance7batch_normalization_6/FusedBatchNormV3:batch_variance:08^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_6/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_6/AssignNewValue_1
re_lu_6/Relu6Relu6*batch_normalization_6/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_6/Relu6σ
+depthwise_conv2d_3/depthwise/ReadVariableOpReadVariableOpOdepthwise_conv2d_3_depthwise_readvariableop_depthwise_conv2d_3_depthwise_kernel*'
_output_shapes
:*
dtype02-
+depthwise_conv2d_3/depthwise/ReadVariableOp‘
"depthwise_conv2d_3/depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            2$
"depthwise_conv2d_3/depthwise/Shape©
*depthwise_conv2d_3/depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2,
*depthwise_conv2d_3/depthwise/dilation_rate
depthwise_conv2d_3/depthwiseDepthwiseConv2dNativere_lu_6/Relu6:activations:03depthwise_conv2d_3/depthwise/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
depthwise_conv2d_3/depthwiseΚ
$batch_normalization_7/ReadVariableOpReadVariableOp@batch_normalization_7_readvariableop_batch_normalization_7_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_7/ReadVariableOpΟ
&batch_normalization_7/ReadVariableOp_1ReadVariableOpAbatch_normalization_7_readvariableop_1_batch_normalization_7_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_7/ReadVariableOp_1
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1χ
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3%depthwise_conv2d_3/depthwise:output:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_7/FusedBatchNormV3΅
$batch_normalization_7/AssignNewValueAssignVariableOpWbatch_normalization_7_fusedbatchnormv3_readvariableop_batch_normalization_7_moving_mean3batch_normalization_7/FusedBatchNormV3:batch_mean:06^batch_normalization_7/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp/batch_normalization_7/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_7/AssignNewValueΛ
&batch_normalization_7/AssignNewValue_1AssignVariableOp]batch_normalization_7_fusedbatchnormv3_readvariableop_1_batch_normalization_7_moving_variance7batch_normalization_7/FusedBatchNormV3:batch_variance:08^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_7/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_7/AssignNewValue_1
re_lu_7/Relu6Relu6*batch_normalization_7/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_7/Relu6Ή
conv2d_4/Conv2D/ReadVariableOpReadVariableOp.conv2d_4_conv2d_readvariableop_conv2d_4_kernel*(
_output_shapes
:*
dtype02 
conv2d_4/Conv2D/ReadVariableOpΛ
conv2d_4/Conv2DConv2Dre_lu_7/Relu6:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
conv2d_4/Conv2DΚ
$batch_normalization_8/ReadVariableOpReadVariableOp@batch_normalization_8_readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02&
$batch_normalization_8/ReadVariableOpΟ
&batch_normalization_8/ReadVariableOp_1ReadVariableOpAbatch_normalization_8_readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02(
&batch_normalization_8/ReadVariableOp_1
5batch_normalization_8/FusedBatchNormV3/ReadVariableOpReadVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype027
5batch_normalization_8/FusedBatchNormV3/ReadVariableOp
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype029
7batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1κ
&batch_normalization_8/FusedBatchNormV3FusedBatchNormV3conv2d_4/Conv2D:output:0,batch_normalization_8/ReadVariableOp:value:0.batch_normalization_8/ReadVariableOp_1:value:0=batch_normalization_8/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*G
_output_shapes5
3::::::*
epsilon%o:*
exponential_avg_factor%
Χ#<2(
&batch_normalization_8/FusedBatchNormV3΅
$batch_normalization_8/AssignNewValueAssignVariableOpWbatch_normalization_8_fusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean3batch_normalization_8/FusedBatchNormV3:batch_mean:06^batch_normalization_8/FusedBatchNormV3/ReadVariableOp*j
_class`
^\loc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp/batch_normalization_8/moving_mean*
_output_shapes
 *
dtype02&
$batch_normalization_8/AssignNewValueΛ
&batch_normalization_8/AssignNewValue_1AssignVariableOp]batch_normalization_8_fusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance7batch_normalization_8/FusedBatchNormV3:batch_variance:08^batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1*p
_classf
dbloc:@batch_normalization_8/FusedBatchNormV3/ReadVariableOp_1/batch_normalization_8/moving_variance*
_output_shapes
 *
dtype02(
&batch_normalization_8/AssignNewValue_1
re_lu_8/Relu6Relu6*batch_normalization_8/FusedBatchNormV3:y:0*
T0*'
_output_shapes
:2
re_lu_8/Relu6³
/global_average_pooling2d/Mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"      21
/global_average_pooling2d/Mean/reduction_indicesΗ
global_average_pooling2d/MeanMeanre_lu_8/Relu6:activations:08global_average_pooling2d/Mean/reduction_indices:output:0*
T0*
_output_shapes
:	2
global_average_pooling2d/Means
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
dropout/dropout/Const£
dropout/dropout/MulMul&global_average_pooling2d/Mean:output:0dropout/dropout/Const:output:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul
dropout/dropout/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      2
dropout/dropout/ShapeΔ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*
_output_shapes
:	*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ΝΜL>2 
dropout/dropout/GreaterEqual/yΦ
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*
_output_shapes
:	2
dropout/dropout/GreaterEqual
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	2
dropout/dropout/Cast
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*
_output_shapes
:	2
dropout/dropout/Mul_1€
dense/MatMul/ReadVariableOpReadVariableOp(dense_matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/MatMul 
dense/BiasAdd/ReadVariableOpReadVariableOp'dense_biasadd_readvariableop_dense_bias*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*
_output_shapes

:2
dense/BiasAdd­
IdentityIdentitydense/BiasAdd:output:0#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_1%^batch_normalization_1/AssignNewValue'^batch_normalization_1/AssignNewValue_1%^batch_normalization_2/AssignNewValue'^batch_normalization_2/AssignNewValue_1%^batch_normalization_3/AssignNewValue'^batch_normalization_3/AssignNewValue_1%^batch_normalization_4/AssignNewValue'^batch_normalization_4/AssignNewValue_1%^batch_normalization_5/AssignNewValue'^batch_normalization_5/AssignNewValue_1%^batch_normalization_6/AssignNewValue'^batch_normalization_6/AssignNewValue_1%^batch_normalization_7/AssignNewValue'^batch_normalization_7/AssignNewValue_1%^batch_normalization_8/AssignNewValue'^batch_normalization_8/AssignNewValue_1*
T0*
_output_shapes

:2

Identity"
identityIdentity:output:0*ί
_input_shapesΝ
Κ:1(:::::::::::::::::::::::::::::::::::::::::::::::2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12L
$batch_normalization_1/AssignNewValue$batch_normalization_1/AssignNewValue2P
&batch_normalization_1/AssignNewValue_1&batch_normalization_1/AssignNewValue_12L
$batch_normalization_2/AssignNewValue$batch_normalization_2/AssignNewValue2P
&batch_normalization_2/AssignNewValue_1&batch_normalization_2/AssignNewValue_12L
$batch_normalization_3/AssignNewValue$batch_normalization_3/AssignNewValue2P
&batch_normalization_3/AssignNewValue_1&batch_normalization_3/AssignNewValue_12L
$batch_normalization_4/AssignNewValue$batch_normalization_4/AssignNewValue2P
&batch_normalization_4/AssignNewValue_1&batch_normalization_4/AssignNewValue_12L
$batch_normalization_5/AssignNewValue$batch_normalization_5/AssignNewValue2P
&batch_normalization_5/AssignNewValue_1&batch_normalization_5/AssignNewValue_12L
$batch_normalization_6/AssignNewValue$batch_normalization_6/AssignNewValue2P
&batch_normalization_6/AssignNewValue_1&batch_normalization_6/AssignNewValue_12L
$batch_normalization_7/AssignNewValue$batch_normalization_7/AssignNewValue2P
&batch_normalization_7/AssignNewValue_1&batch_normalization_7/AssignNewValue_12L
$batch_normalization_8/AssignNewValue$batch_normalization_8/AssignNewValue2P
&batch_normalization_8/AssignNewValue_1&batch_normalization_8/AssignNewValue_1:T P
+
_output_shapes
:?????????1(
!
_user_specified_name	input_1
?

O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_1887

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_2/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_2/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?
l
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3881

inputs
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*
_cloned(*&
_output_shapes
:1(2

ExpandDimsf
IdentityIdentityExpandDims:output:0*
T0*&
_output_shapes
:1(2

Identity"
identityIdentity:output:0*!
_input_shapes
:1(:J F
"
_output_shapes
:1(
 
_user_specified_nameinputs
©
­
?__inference_dense_layer_call_and_return_conditional_losses_5050

inputs&
"matmul_readvariableop_dense_kernel%
!biasadd_readvariableop_dense_bias
identity
MatMul/ReadVariableOpReadVariableOp"matmul_readvariableop_dense_kernel*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*
_output_shapes

:2
MatMul
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_dense_bias*
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
:	:::G C

_output_shapes
:	
 
_user_specified_nameinputs
λ

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1192

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1κ
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1¦
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
―
]
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4510

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
:@2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*%
_input_shapes
:@:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
?

O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_1967

inputs.
*readvariableop_batch_normalization_3_gamma/
+readvariableop_1_batch_normalization_3_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_3_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_3_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_3_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_3/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_3_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_3/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
: 
 
_user_specified_nameinputs
?

O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_2056

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity’AssignNewValue’AssignNewValue_1
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Ο
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.:@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
Χ#<2
FusedBatchNormV3±
AssignNewValueAssignVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*T
_classJ
HFloc:@FusedBatchNormV3/ReadVariableOp/batch_normalization_4/moving_mean*
_output_shapes
 *
dtype02
AssignNewValueΗ
AssignNewValue_1AssignVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*Z
_classP
NLloc:@FusedBatchNormV3/ReadVariableOp_1/batch_normalization_4/moving_variance*
_output_shapes
 *
dtype02
AssignNewValue_1
IdentityIdentityFusedBatchNormV3:y:0^AssignNewValue^AssignNewValue_1*
T0*&
_output_shapes
:@2

Identity"
identityIdentity:output:0*5
_input_shapes$
":@::::2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_1:N J
&
_output_shapes
:@
 
_user_specified_nameinputs
Δ

@__inference_conv2d_layer_call_and_return_conditional_losses_3893

inputs'
#conv2d_readvariableop_conv2d_kernel
identity
Conv2D/ReadVariableOpReadVariableOp#conv2d_readvariableop_conv2d_kernel*&
_output_shapes
:( *
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*&
_output_shapes
: *
paddingVALID*
strides
2
Conv2Db
IdentityIdentityConv2D:output:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*)
_input_shapes
:1(::N J
&
_output_shapes
:1(
 
_user_specified_nameinputs
―
]
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4379

inputs
identityP
Relu6Relu6inputs*
T0*&
_output_shapes
: 2
Relu6f
IdentityIdentityRelu6:activations:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*%
_input_shapes
: :N J
&
_output_shapes
: 
 
_user_specified_nameinputs
Ϋ
γ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4985

inputs.
*readvariableop_batch_normalization_8_gamma/
+readvariableop_1_batch_normalization_8_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_8_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_8_gamma*
_output_shapes	
:*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_8_beta*
_output_shapes	
:*
dtype02
ReadVariableOp_1Α
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_8_moving_mean*
_output_shapes	
:*
dtype02!
FusedBatchNormV3/ReadVariableOpΛ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_8_moving_variance*
_output_shapes	
:*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,???????????????????????????:::::*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*B
_output_shapes0
.:,???????????????????????????2

Identity"
identityIdentity:output:0*Q
_input_shapes@
>:,???????????????????????????:::::j f
B
_output_shapes0
.:,???????????????????????????
 
_user_specified_nameinputs
ΌΚ
³
 __inference__traced_restore_5372
file_prefix"
assignvariableop_conv2d_kernel0
,assignvariableop_1_batch_normalization_gamma/
+assignvariableop_2_batch_normalization_beta6
2assignvariableop_3_batch_normalization_moving_mean:
6assignvariableop_4_batch_normalization_moving_variance8
4assignvariableop_5_depthwise_conv2d_depthwise_kernel2
.assignvariableop_6_batch_normalization_1_gamma1
-assignvariableop_7_batch_normalization_1_beta8
4assignvariableop_8_batch_normalization_1_moving_mean<
8assignvariableop_9_batch_normalization_1_moving_variance'
#assignvariableop_10_conv2d_1_kernel3
/assignvariableop_11_batch_normalization_2_gamma2
.assignvariableop_12_batch_normalization_2_beta9
5assignvariableop_13_batch_normalization_2_moving_mean=
9assignvariableop_14_batch_normalization_2_moving_variance;
7assignvariableop_15_depthwise_conv2d_1_depthwise_kernel3
/assignvariableop_16_batch_normalization_3_gamma2
.assignvariableop_17_batch_normalization_3_beta9
5assignvariableop_18_batch_normalization_3_moving_mean=
9assignvariableop_19_batch_normalization_3_moving_variance'
#assignvariableop_20_conv2d_2_kernel3
/assignvariableop_21_batch_normalization_4_gamma2
.assignvariableop_22_batch_normalization_4_beta9
5assignvariableop_23_batch_normalization_4_moving_mean=
9assignvariableop_24_batch_normalization_4_moving_variance;
7assignvariableop_25_depthwise_conv2d_2_depthwise_kernel3
/assignvariableop_26_batch_normalization_5_gamma2
.assignvariableop_27_batch_normalization_5_beta9
5assignvariableop_28_batch_normalization_5_moving_mean=
9assignvariableop_29_batch_normalization_5_moving_variance'
#assignvariableop_30_conv2d_3_kernel3
/assignvariableop_31_batch_normalization_6_gamma2
.assignvariableop_32_batch_normalization_6_beta9
5assignvariableop_33_batch_normalization_6_moving_mean=
9assignvariableop_34_batch_normalization_6_moving_variance;
7assignvariableop_35_depthwise_conv2d_3_depthwise_kernel3
/assignvariableop_36_batch_normalization_7_gamma2
.assignvariableop_37_batch_normalization_7_beta9
5assignvariableop_38_batch_normalization_7_moving_mean=
9assignvariableop_39_batch_normalization_7_moving_variance'
#assignvariableop_40_conv2d_4_kernel3
/assignvariableop_41_batch_normalization_8_gamma2
.assignvariableop_42_batch_normalization_8_beta9
5assignvariableop_43_batch_normalization_8_moving_mean=
9assignvariableop_44_batch_normalization_8_moving_variance$
 assignvariableop_45_dense_kernel"
assignvariableop_46_dense_bias
identity_48’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_25’AssignVariableOp_26’AssignVariableOp_27’AssignVariableOp_28’AssignVariableOp_29’AssignVariableOp_3’AssignVariableOp_30’AssignVariableOp_31’AssignVariableOp_32’AssignVariableOp_33’AssignVariableOp_34’AssignVariableOp_35’AssignVariableOp_36’AssignVariableOp_37’AssignVariableOp_38’AssignVariableOp_39’AssignVariableOp_4’AssignVariableOp_40’AssignVariableOp_41’AssignVariableOp_42’AssignVariableOp_43’AssignVariableOp_44’AssignVariableOp_45’AssignVariableOp_46’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*
valueB0B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-2/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-6/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-10/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEBAlayer_with_weights-14/depthwise_kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesξ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:0*
dtype0*s
valuejBh0B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Φ
_output_shapesΓ
ΐ::::::::::::::::::::::::::::::::::::::::::::::::*>
dtypes4
2202
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_batch_normalization_gammaIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2°
AssignVariableOp_2AssignVariableOp+assignvariableop_2_batch_normalization_betaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3·
AssignVariableOp_3AssignVariableOp2assignvariableop_3_batch_normalization_moving_meanIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4»
AssignVariableOp_4AssignVariableOp6assignvariableop_4_batch_normalization_moving_varianceIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ή
AssignVariableOp_5AssignVariableOp4assignvariableop_5_depthwise_conv2d_depthwise_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6³
AssignVariableOp_6AssignVariableOp.assignvariableop_6_batch_normalization_1_gammaIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7²
AssignVariableOp_7AssignVariableOp-assignvariableop_7_batch_normalization_1_betaIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ή
AssignVariableOp_8AssignVariableOp4assignvariableop_8_batch_normalization_1_moving_meanIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9½
AssignVariableOp_9AssignVariableOp8assignvariableop_9_batch_normalization_1_moving_varianceIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10«
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11·
AssignVariableOp_11AssignVariableOp/assignvariableop_11_batch_normalization_2_gammaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ά
AssignVariableOp_12AssignVariableOp.assignvariableop_12_batch_normalization_2_betaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13½
AssignVariableOp_13AssignVariableOp5assignvariableop_13_batch_normalization_2_moving_meanIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Α
AssignVariableOp_14AssignVariableOp9assignvariableop_14_batch_normalization_2_moving_varianceIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15Ώ
AssignVariableOp_15AssignVariableOp7assignvariableop_15_depthwise_conv2d_1_depthwise_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16·
AssignVariableOp_16AssignVariableOp/assignvariableop_16_batch_normalization_3_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17Ά
AssignVariableOp_17AssignVariableOp.assignvariableop_17_batch_normalization_3_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18½
AssignVariableOp_18AssignVariableOp5assignvariableop_18_batch_normalization_3_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Α
AssignVariableOp_19AssignVariableOp9assignvariableop_19_batch_normalization_3_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp#assignvariableop_20_conv2d_2_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21·
AssignVariableOp_21AssignVariableOp/assignvariableop_21_batch_normalization_4_gammaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22Ά
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batch_normalization_4_betaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23½
AssignVariableOp_23AssignVariableOp5assignvariableop_23_batch_normalization_4_moving_meanIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24Α
AssignVariableOp_24AssignVariableOp9assignvariableop_24_batch_normalization_4_moving_varianceIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25Ώ
AssignVariableOp_25AssignVariableOp7assignvariableop_25_depthwise_conv2d_2_depthwise_kernelIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26·
AssignVariableOp_26AssignVariableOp/assignvariableop_26_batch_normalization_5_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27Ά
AssignVariableOp_27AssignVariableOp.assignvariableop_27_batch_normalization_5_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28½
AssignVariableOp_28AssignVariableOp5assignvariableop_28_batch_normalization_5_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29Α
AssignVariableOp_29AssignVariableOp9assignvariableop_29_batch_normalization_5_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30«
AssignVariableOp_30AssignVariableOp#assignvariableop_30_conv2d_3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31·
AssignVariableOp_31AssignVariableOp/assignvariableop_31_batch_normalization_6_gammaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32Ά
AssignVariableOp_32AssignVariableOp.assignvariableop_32_batch_normalization_6_betaIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33½
AssignVariableOp_33AssignVariableOp5assignvariableop_33_batch_normalization_6_moving_meanIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34Α
AssignVariableOp_34AssignVariableOp9assignvariableop_34_batch_normalization_6_moving_varianceIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35Ώ
AssignVariableOp_35AssignVariableOp7assignvariableop_35_depthwise_conv2d_3_depthwise_kernelIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36·
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_7_gammaIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37Ά
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_7_betaIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38½
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_7_moving_meanIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39Α
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_7_moving_varianceIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40«
AssignVariableOp_40AssignVariableOp#assignvariableop_40_conv2d_4_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41·
AssignVariableOp_41AssignVariableOp/assignvariableop_41_batch_normalization_8_gammaIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42Ά
AssignVariableOp_42AssignVariableOp.assignvariableop_42_batch_normalization_8_betaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43½
AssignVariableOp_43AssignVariableOp5assignvariableop_43_batch_normalization_8_moving_meanIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44Α
AssignVariableOp_44AssignVariableOp9assignvariableop_44_batch_normalization_8_moving_varianceIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¨
AssignVariableOp_45AssignVariableOp assignvariableop_45_dense_kernelIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¦
AssignVariableOp_46AssignVariableOpassignvariableop_46_dense_biasIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_469
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpθ
Identity_47Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_47Ϋ
Identity_48IdentityIdentity_47:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_48"#
identity_48Identity_48:output:0*Σ
_input_shapesΑ
Ύ: :::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462(
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
Ο
γ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4238

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+??????????????????????????? : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+??????????????????????????? :::::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
Ο
γ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_1219

inputs.
*readvariableop_batch_normalization_4_gamma/
+readvariableop_1_batch_normalization_4_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_4_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_4_gamma*
_output_shapes
:@*
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_4_beta*
_output_shapes
:@*
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_4_moving_mean*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_4_moving_variance*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1ά
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o:*
is_training( 2
FusedBatchNormV3
IdentityIdentityFusedBatchNormV3:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+???????????????????????????@:::::i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
Μ

B__inference_conv2d_3_layer_call_and_return_conditional_losses_4640

inputs)
%conv2d_readvariableop_conv2d_3_kernel
identity
Conv2D/ReadVariableOpReadVariableOp%conv2d_readvariableop_conv2d_3_kernel*'
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*'
_output_shapes
:*
paddingSAME*
strides
2
Conv2Dc
IdentityIdentityConv2D:output:0*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*)
_input_shapes
:@::N J
&
_output_shapes
:@
 
_user_specified_nameinputs
ε

4__inference_batch_normalization_7_layer_call_fn_4818

inputs
batch_normalization_7_gamma
batch_normalization_7_beta%
!batch_normalization_7_moving_mean)
%batch_normalization_7_moving_variance
identity’StatefulPartitionedCallς
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_7_gammabatch_normalization_7_beta!batch_normalization_7_moving_mean%batch_normalization_7_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23232
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
ε

4__inference_batch_normalization_8_layer_call_fn_4949

inputs
batch_normalization_8_gamma
batch_normalization_8_beta%
!batch_normalization_8_moving_mean)
%batch_normalization_8_moving_variance
identity’StatefulPartitionedCallς
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_8_gammabatch_normalization_8_beta!batch_normalization_8_moving_mean%batch_normalization_8_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_24122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
β
γ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4184

inputs.
*readvariableop_batch_normalization_2_gamma/
+readvariableop_1_batch_normalization_2_betaE
Afusedbatchnormv3_readvariableop_batch_normalization_2_moving_meanK
Gfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance
identity
ReadVariableOpReadVariableOp*readvariableop_batch_normalization_2_gamma*
_output_shapes
: *
dtype02
ReadVariableOp
ReadVariableOp_1ReadVariableOp+readvariableop_1_batch_normalization_2_beta*
_output_shapes
: *
dtype02
ReadVariableOp_1ΐ
FusedBatchNormV3/ReadVariableOpReadVariableOpAfusedbatchnormv3_readvariableop_batch_normalization_2_moving_mean*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOpΚ
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOpGfusedbatchnormv3_readvariableop_1_batch_normalization_2_moving_variance*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1Α
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*B
_output_shapes0
.: : : : : :*
epsilon%o:*
is_training( 2
FusedBatchNormV3g
IdentityIdentityFusedBatchNormV3:y:0*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": :::::N J
&
_output_shapes
: 
 
_user_specified_nameinputs
ε

4__inference_batch_normalization_6_layer_call_fn_4754

inputs
batch_normalization_6_gamma
batch_normalization_6_beta%
!batch_normalization_6_moving_mean)
%batch_normalization_6_moving_variance
identity’StatefulPartitionedCallς
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_6_gammabatch_normalization_6_beta!batch_normalization_6_moving_mean%batch_normalization_6_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:
 
_user_specified_nameinputs
	
¨
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_828

inputs>
:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel
identity·
depthwise/ReadVariableOpReadVariableOp:depthwise_readvariableop_depthwise_conv2d_depthwise_kernel*&
_output_shapes
: *
dtype02
depthwise/ReadVariableOp{
depthwise/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"             2
depthwise/Shape
depthwise/dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
depthwise/dilation_rateΞ
	depthwiseDepthwiseConv2dNativeinputs depthwise/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingVALID*
strides
2
	depthwise
IdentityIdentitydepthwise:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*D
_input_shapes3
1:+??????????????????????????? ::i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
α

4__inference_batch_normalization_1_layer_call_fn_4071

inputs
batch_normalization_1_gamma
batch_normalization_1_beta%
!batch_normalization_1_moving_mean)
%batch_normalization_1_moving_variance
identity’StatefulPartitionedCallρ
StatefulPartitionedCallStatefulPartitionedCallinputsbatch_normalization_1_gammabatch_normalization_1_beta!batch_normalization_1_moving_mean%batch_normalization_1_moving_variance*
Tin	
2*
Tout
2*
_collective_manager_ids
 *&
_output_shapes
: *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *X
fSRQ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_18162
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*&
_output_shapes
: 2

Identity"
identityIdentity:output:0*5
_input_shapes$
": ::::22
StatefulPartitionedCallStatefulPartitionedCall:N J
&
_output_shapes
: 
 
_user_specified_nameinputs"ΈL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
6
input_1+
serving_default_input_1:01(0
dense'
StatefulPartitionedCall:0tensorflow/serving/predict:Λπ

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
layer-10
layer_with_weights-6
layer-11
layer_with_weights-7
layer-12
layer-13
layer_with_weights-8
layer-14
layer_with_weights-9
layer-15
layer-16
layer_with_weights-10
layer-17
layer_with_weights-11
layer-18
layer-19
layer_with_weights-12
layer-20
layer_with_weights-13
layer-21
layer-22
layer_with_weights-14
layer-23
layer_with_weights-15
layer-24
layer-25
layer_with_weights-16
layer-26
layer_with_weights-17
layer-27
layer-28
layer-29
layer-30
 layer_with_weights-18
 layer-31
!trainable_variables
"	variables
#regularization_losses
$	keras_api
%
signatures
ϊ_default_save_signature
ϋ__call__
+ό&call_and_return_all_conditional_losses"«
_tf_keras_network{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 49, 40]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "TensorFlowOpLayer", "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}, "name": "tf_op_layer_ExpandDims", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["tf_op_layer_ExpandDims", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d", "inbound_nodes": [[["re_lu", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["depthwise_conv2d", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_1", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["re_lu_1", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_2", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_1", "inbound_nodes": [[["re_lu_2", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["depthwise_conv2d_1", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_3", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["re_lu_3", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_4", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_2", "inbound_nodes": [[["re_lu_4", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["depthwise_conv2d_2", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_5", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_3", "inbound_nodes": [[["re_lu_5", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_6", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "DepthwiseConv2D", "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "name": "depthwise_conv2d_3", "inbound_nodes": [[["re_lu_6", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["depthwise_conv2d_3", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_7", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_4", "inbound_nodes": [[["re_lu_7", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_8", "inbound_nodes": [[["conv2d_4", 0, 0, {}]]]}, {"class_name": "ReLU", "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}, "name": "re_lu_8", "inbound_nodes": [[["batch_normalization_8", 0, 0, {}]]]}, {"class_name": "GlobalAveragePooling2D", "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "global_average_pooling2d", "inbound_nodes": [[["re_lu_8", 0, 0, {}]]]}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "name": "dropout", "inbound_nodes": [[["global_average_pooling2d", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["dropout", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense", 0, 0]]}}}
ν"κ
_tf_keras_input_layerΚ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [1, 49, 40]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}

&trainable_variables
'	variables
(regularization_losses
)	keras_api
ύ__call__
+ώ&call_and_return_all_conditional_losses"ω
_tf_keras_layerί{"class_name": "TensorFlowOpLayer", "name": "tf_op_layer_ExpandDims", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "ExpandDims", "trainable": true, "dtype": "float32", "node_def": {"name": "ExpandDims", "op": "ExpandDims", "input": ["input_1", "ExpandDims/dim"], "attr": {"Tdim": {"type": "DT_INT32"}, "T": {"type": "DT_FLOAT"}}}, "constants": {"1": 2}}}
	

*kernel
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?__call__
+&call_and_return_all_conditional_losses"ό
_tf_keras_layerβ{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 40}}}}
δ
/axis
	0gamma
1beta
2moving_mean
3moving_variance
4trainable_variables
5	variables
6regularization_losses
7	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerτ{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
θ
8trainable_variables
9	variables
:regularization_losses
;	keras_api
__call__
+&call_and_return_all_conditional_losses"Χ
_tf_keras_layer½{"class_name": "ReLU", "name": "re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Ρ	
<depthwise_kernel
=trainable_variables
>	variables
?regularization_losses
@	keras_api
__call__
+&call_and_return_all_conditional_losses"ͺ
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
θ
Aaxis
	Bgamma
Cbeta
Dmoving_mean
Emoving_variance
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerψ{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
μ
Jtrainable_variables
K	variables
Lregularization_losses
M	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_1", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
	

Nkernel
Otrainable_variables
P	variables
Qregularization_losses
R	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerε{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
θ
Saxis
	Tgamma
Ubeta
Vmoving_mean
Wmoving_variance
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerψ{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
μ
\trainable_variables
]	variables
^regularization_losses
_	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_2", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Υ	
`depthwise_kernel
atrainable_variables
b	variables
cregularization_losses
d	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_1", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
θ
eaxis
	fgamma
gbeta
hmoving_mean
imoving_variance
jtrainable_variables
k	variables
lregularization_losses
m	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerψ{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
μ
ntrainable_variables
o	variables
pregularization_losses
q	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_3", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
	

rkernel
strainable_variables
t	variables
uregularization_losses
v	keras_api
__call__
+&call_and_return_all_conditional_losses"?
_tf_keras_layerε{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}}
θ
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}	variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layerψ{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
π
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_4", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Ω	
depthwise_kernel
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"­
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_2", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
ρ
	axis

gamma
	beta
moving_mean
moving_variance
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layerψ{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
π
trainable_variables
	variables
regularization_losses
	keras_api
‘__call__
+’&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_5", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
’	
kernel
trainable_variables
	variables
regularization_losses
	keras_api
£__call__
+€&call_and_return_all_conditional_losses"
_tf_keras_layerζ{"class_name": "Conv2D", "name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}}}
ς
	axis

gamma
	beta
moving_mean
moving_variance
 trainable_variables
‘	variables
’regularization_losses
£	keras_api
₯__call__
+¦&call_and_return_all_conditional_losses"
_tf_keras_layerω{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
π
€trainable_variables
₯	variables
¦regularization_losses
§	keras_api
§__call__
+¨&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_6", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
Ϊ	
¨depthwise_kernel
©trainable_variables
ͺ	variables
«regularization_losses
¬	keras_api
©__call__
+ͺ&call_and_return_all_conditional_losses"?
_tf_keras_layer{"class_name": "DepthwiseConv2D", "name": "depthwise_conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "depthwise_conv2d_3", "trainable": true, "dtype": "float32", "kernel_size": {"class_name": "__tuple__", "items": [3, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "bias_regularizer": null, "activity_regularizer": null, "bias_constraint": null, "depth_multiplier": 1, "depthwise_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "depthwise_regularizer": null, "depthwise_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}}
ς
	­axis

?gamma
	―beta
°moving_mean
±moving_variance
²trainable_variables
³	variables
΄regularization_losses
΅	keras_api
«__call__
+¬&call_and_return_all_conditional_losses"
_tf_keras_layerω{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
π
Άtrainable_variables
·	variables
Έregularization_losses
Ή	keras_api
­__call__
+?&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_7", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}
£	
Ίkernel
»trainable_variables
Ό	variables
½regularization_losses
Ύ	keras_api
―__call__
+°&call_and_return_all_conditional_losses"
_tf_keras_layerη{"class_name": "Conv2D", "name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}}
ς
	Ώaxis

ΐgamma
	Αbeta
Βmoving_mean
Γmoving_variance
Δtrainable_variables
Ε	variables
Ζregularization_losses
Η	keras_api
±__call__
+²&call_and_return_all_conditional_losses"
_tf_keras_layerω{"class_name": "BatchNormalization", "name": "batch_normalization_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_8", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": 1, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 128}}}}
π
Θtrainable_variables
Ι	variables
Κregularization_losses
Λ	keras_api
³__call__
+΄&call_and_return_all_conditional_losses"Ϋ
_tf_keras_layerΑ{"class_name": "ReLU", "name": "re_lu_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "re_lu_8", "trainable": true, "dtype": "float32", "max_value": 6.0, "negative_slope": 0.0, "threshold": 0.0}}

Μtrainable_variables
Ν	variables
Ξregularization_losses
Ο	keras_api
΅__call__
+Ά&call_and_return_all_conditional_losses"
_tf_keras_layerκ{"class_name": "GlobalAveragePooling2D", "name": "global_average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "global_average_pooling2d", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
η
Πtrainable_variables
Ρ	variables
?regularization_losses
Σ	keras_api
·__call__
+Έ&call_and_return_all_conditional_losses"?
_tf_keras_layerΈ{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
?
Τkernel
	Υbias
Φtrainable_variables
Χ	variables
Ψregularization_losses
Ω	keras_api
Ή__call__
+Ί&call_and_return_all_conditional_losses"
_tf_keras_layerη{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 14, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}}

*0
01
12
<3
B4
C5
N6
T7
U8
`9
f10
g11
r12
x13
y14
15
16
17
18
19
20
¨21
?22
―23
Ί24
ΐ25
Α26
Τ27
Υ28"
trackable_list_wrapper
€
*0
01
12
23
34
<5
B6
C7
D8
E9
N10
T11
U12
V13
W14
`15
f16
g17
h18
i19
r20
x21
y22
z23
{24
25
26
27
28
29
30
31
32
33
34
¨35
?36
―37
°38
±39
Ί40
ΐ41
Α42
Β43
Γ44
Τ45
Υ46"
trackable_list_wrapper
 "
trackable_list_wrapper
Σ
Ϊmetrics
Ϋlayers
!trainable_variables
άlayer_metrics
έnon_trainable_variables
 ήlayer_regularization_losses
"	variables
#regularization_losses
ϋ__call__
ϊ_default_save_signature
+ό&call_and_return_all_conditional_losses
'ό"call_and_return_conditional_losses"
_generic_user_object
-
»serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ίmetrics
ΰlayers
&trainable_variables
αlayer_metrics
βnon_trainable_variables
 γlayer_regularization_losses
'	variables
(regularization_losses
ύ__call__
+ώ&call_and_return_all_conditional_losses
'ώ"call_and_return_conditional_losses"
_generic_user_object
':%( 2conv2d/kernel
'
*0"
trackable_list_wrapper
'
*0"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
δmetrics
εlayers
+trainable_variables
ζlayer_metrics
ηnon_trainable_variables
 θlayer_regularization_losses
,	variables
-regularization_losses
?__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
.
00
11"
trackable_list_wrapper
<
00
11
22
33"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ιmetrics
κlayers
4trainable_variables
λlayer_metrics
μnon_trainable_variables
 νlayer_regularization_losses
5	variables
6regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ξmetrics
οlayers
8trainable_variables
πlayer_metrics
ρnon_trainable_variables
 ςlayer_regularization_losses
9	variables
:regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
;:9 2!depthwise_conv2d/depthwise_kernel
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
σmetrics
τlayers
=trainable_variables
υlayer_metrics
φnon_trainable_variables
 χlayer_regularization_losses
>	variables
?regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
.
B0
C1"
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ψmetrics
ωlayers
Ftrainable_variables
ϊlayer_metrics
ϋnon_trainable_variables
 όlayer_regularization_losses
G	variables
Hregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
ύmetrics
ώlayers
Jtrainable_variables
?layer_metrics
non_trainable_variables
 layer_regularization_losses
K	variables
Lregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):'  2conv2d_1/kernel
'
N0"
trackable_list_wrapper
'
N0"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
layers
Otrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
P	variables
Qregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_2/gamma
(:& 2batch_normalization_2/beta
1:/  (2!batch_normalization_2/moving_mean
5:3  (2%batch_normalization_2/moving_variance
.
T0
U1"
trackable_list_wrapper
<
T0
U1
V2
W3"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
layers
Xtrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
Y	variables
Zregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
layers
\trainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
]	variables
^regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
=:; 2#depthwise_conv2d_1/depthwise_kernel
'
`0"
trackable_list_wrapper
'
`0"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
layers
atrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
b	variables
cregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_3/gamma
(:& 2batch_normalization_3/beta
1:/  (2!batch_normalization_3/moving_mean
5:3  (2%batch_normalization_3/moving_variance
.
f0
g1"
trackable_list_wrapper
<
f0
g1
h2
i3"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
layers
jtrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
k	variables
lregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
΅
metrics
layers
ntrainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
o	variables
pregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_2/kernel
'
r0"
trackable_list_wrapper
'
r0"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
 metrics
‘layers
strainable_variables
’layer_metrics
£non_trainable_variables
 €layer_regularization_losses
t	variables
uregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
.
x0
y1"
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
 "
trackable_list_wrapper
΅
₯metrics
¦layers
|trainable_variables
§layer_metrics
¨non_trainable_variables
 ©layer_regularization_losses
}	variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ͺmetrics
«layers
trainable_variables
¬layer_metrics
­non_trainable_variables
 ?layer_regularization_losses
	variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
=:;@2#depthwise_conv2d_2/depthwise_kernel
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
―metrics
°layers
trainable_variables
±layer_metrics
²non_trainable_variables
 ³layer_regularization_losses
	variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
΄metrics
΅layers
trainable_variables
Άlayer_metrics
·non_trainable_variables
 Έlayer_regularization_losses
	variables
regularization_losses
__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ήmetrics
Ίlayers
trainable_variables
»layer_metrics
Όnon_trainable_variables
 ½layer_regularization_losses
	variables
regularization_losses
‘__call__
+’&call_and_return_all_conditional_losses
'’"call_and_return_conditional_losses"
_generic_user_object
*:(@2conv2d_3/kernel
(
0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Ύmetrics
Ώlayers
trainable_variables
ΐlayer_metrics
Αnon_trainable_variables
 Βlayer_regularization_losses
	variables
regularization_losses
£__call__
+€&call_and_return_all_conditional_losses
'€"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_6/gamma
):'2batch_normalization_6/beta
2:0 (2!batch_normalization_6/moving_mean
6:4 (2%batch_normalization_6/moving_variance
0
0
1"
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Γmetrics
Δlayers
 trainable_variables
Εlayer_metrics
Ζnon_trainable_variables
 Ηlayer_regularization_losses
‘	variables
’regularization_losses
₯__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Θmetrics
Ιlayers
€trainable_variables
Κlayer_metrics
Λnon_trainable_variables
 Μlayer_regularization_losses
₯	variables
¦regularization_losses
§__call__
+¨&call_and_return_all_conditional_losses
'¨"call_and_return_conditional_losses"
_generic_user_object
>:<2#depthwise_conv2d_3/depthwise_kernel
(
¨0"
trackable_list_wrapper
(
¨0"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Νmetrics
Ξlayers
©trainable_variables
Οlayer_metrics
Πnon_trainable_variables
 Ρlayer_regularization_losses
ͺ	variables
«regularization_losses
©__call__
+ͺ&call_and_return_all_conditional_losses
'ͺ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_7/gamma
):'2batch_normalization_7/beta
2:0 (2!batch_normalization_7/moving_mean
6:4 (2%batch_normalization_7/moving_variance
0
?0
―1"
trackable_list_wrapper
@
?0
―1
°2
±3"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
?metrics
Σlayers
²trainable_variables
Τlayer_metrics
Υnon_trainable_variables
 Φlayer_regularization_losses
³	variables
΄regularization_losses
«__call__
+¬&call_and_return_all_conditional_losses
'¬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
Χmetrics
Ψlayers
Άtrainable_variables
Ωlayer_metrics
Ϊnon_trainable_variables
 Ϋlayer_regularization_losses
·	variables
Έregularization_losses
­__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)2conv2d_4/kernel
(
Ί0"
trackable_list_wrapper
(
Ί0"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
άmetrics
έlayers
»trainable_variables
ήlayer_metrics
ίnon_trainable_variables
 ΰlayer_regularization_losses
Ό	variables
½regularization_losses
―__call__
+°&call_and_return_all_conditional_losses
'°"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
*:(2batch_normalization_8/gamma
):'2batch_normalization_8/beta
2:0 (2!batch_normalization_8/moving_mean
6:4 (2%batch_normalization_8/moving_variance
0
ΐ0
Α1"
trackable_list_wrapper
@
ΐ0
Α1
Β2
Γ3"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
αmetrics
βlayers
Δtrainable_variables
γlayer_metrics
δnon_trainable_variables
 εlayer_regularization_losses
Ε	variables
Ζregularization_losses
±__call__
+²&call_and_return_all_conditional_losses
'²"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
ζmetrics
ηlayers
Θtrainable_variables
θlayer_metrics
ιnon_trainable_variables
 κlayer_regularization_losses
Ι	variables
Κregularization_losses
³__call__
+΄&call_and_return_all_conditional_losses
'΄"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
λmetrics
μlayers
Μtrainable_variables
νlayer_metrics
ξnon_trainable_variables
 οlayer_regularization_losses
Ν	variables
Ξregularization_losses
΅__call__
+Ά&call_and_return_all_conditional_losses
'Ά"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
πmetrics
ρlayers
Πtrainable_variables
ςlayer_metrics
σnon_trainable_variables
 τlayer_regularization_losses
Ρ	variables
?regularization_losses
·__call__
+Έ&call_and_return_all_conditional_losses
'Έ"call_and_return_conditional_losses"
_generic_user_object
:	2dense/kernel
:2
dense/bias
0
Τ0
Υ1"
trackable_list_wrapper
0
Τ0
Υ1"
trackable_list_wrapper
 "
trackable_list_wrapper
Έ
υmetrics
φlayers
Φtrainable_variables
χlayer_metrics
ψnon_trainable_variables
 ωlayer_regularization_losses
Χ	variables
Ψregularization_losses
Ή__call__
+Ί&call_and_return_all_conditional_losses
'Ί"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper

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
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31"
trackable_list_wrapper
 "
trackable_dict_wrapper
?
20
31
D2
E3
V4
W5
h6
i7
z8
{9
10
11
12
13
°14
±15
Β16
Γ17"
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
.
20
31"
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
.
D0
E1"
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
.
V0
W1"
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
.
h0
i1"
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
.
z0
{1"
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
0
0
1"
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
0
0
1"
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
0
°0
±1"
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
0
Β0
Γ1"
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
ΰ2έ
__inference__wrapped_model_714Ί
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
input_1?????????1(
ϊ2χ
+__inference_functional_1_layer_call_fn_3875
+__inference_functional_1_layer_call_fn_3342
+__inference_functional_1_layer_call_fn_3394
+__inference_functional_1_layer_call_fn_3823ΐ
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
ζ2γ
F__inference_functional_1_layer_call_and_return_conditional_losses_3290
F__inference_functional_1_layer_call_and_return_conditional_losses_3771
F__inference_functional_1_layer_call_and_return_conditional_losses_3105
F__inference_functional_1_layer_call_and_return_conditional_losses_3586ΐ
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
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3886’
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
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3881’
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
%__inference_conv2d_layer_call_fn_3899’
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
@__inference_conv2d_layer_call_and_return_conditional_losses_3893’
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
2
2__inference_batch_normalization_layer_call_fn_3998
2__inference_batch_normalization_layer_call_fn_4007
2__inference_batch_normalization_layer_call_fn_3953
2__inference_batch_normalization_layer_call_fn_3944΄
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
φ2σ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3917
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3935
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3989
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3971΄
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
$__inference_re_lu_layer_call_fn_4017’
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
?__inference_re_lu_layer_call_and_return_conditional_losses_4012’
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
2
.__inference_depthwise_conv2d_layer_call_fn_832Χ
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
annotationsͺ *7’4
2/+??????????????????????????? 
¨2₯
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_816Χ
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
annotationsͺ *7’4
2/+??????????????????????????? 
2
4__inference_batch_normalization_1_layer_call_fn_4071
4__inference_batch_normalization_1_layer_call_fn_4125
4__inference_batch_normalization_1_layer_call_fn_4116
4__inference_batch_normalization_1_layer_call_fn_4062΄
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
ώ2ϋ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4089
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4053
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4107
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4035΄
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
Π2Ν
&__inference_re_lu_1_layer_call_fn_4135’
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
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4130’
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
'__inference_conv2d_1_layer_call_fn_4148’
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
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4142’
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
2
4__inference_batch_normalization_2_layer_call_fn_4193
4__inference_batch_normalization_2_layer_call_fn_4247
4__inference_batch_normalization_2_layer_call_fn_4256
4__inference_batch_normalization_2_layer_call_fn_4202΄
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
ώ2ϋ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4220
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4184
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4238
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4166΄
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
Π2Ν
&__inference_re_lu_2_layer_call_fn_4266’
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
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4261’
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
2
1__inference_depthwise_conv2d_1_layer_call_fn_1042Χ
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
annotationsͺ *7’4
2/+??????????????????????????? 
«2¨
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1026Χ
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
annotationsͺ *7’4
2/+??????????????????????????? 
2
4__inference_batch_normalization_3_layer_call_fn_4320
4__inference_batch_normalization_3_layer_call_fn_4311
4__inference_batch_normalization_3_layer_call_fn_4365
4__inference_batch_normalization_3_layer_call_fn_4374΄
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
ώ2ϋ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4302
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4338
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4284
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4356΄
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
Π2Ν
&__inference_re_lu_3_layer_call_fn_4384’
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
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4379’
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
'__inference_conv2d_2_layer_call_fn_4397’
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
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4391’
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
2
4__inference_batch_normalization_4_layer_call_fn_4505
4__inference_batch_normalization_4_layer_call_fn_4442
4__inference_batch_normalization_4_layer_call_fn_4451
4__inference_batch_normalization_4_layer_call_fn_4496΄
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
ώ2ϋ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4433
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4415
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4469
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4487΄
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
Π2Ν
&__inference_re_lu_4_layer_call_fn_4515’
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
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4510’
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
2
1__inference_depthwise_conv2d_2_layer_call_fn_1252Χ
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
annotationsͺ *7’4
2/+???????????????????????????@
«2¨
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1236Χ
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
annotationsͺ *7’4
2/+???????????????????????????@
2
4__inference_batch_normalization_5_layer_call_fn_4560
4__inference_batch_normalization_5_layer_call_fn_4623
4__inference_batch_normalization_5_layer_call_fn_4569
4__inference_batch_normalization_5_layer_call_fn_4614΄
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
ώ2ϋ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4605
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4587
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4533
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4551΄
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
Π2Ν
&__inference_re_lu_5_layer_call_fn_4633’
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
A__inference_re_lu_5_layer_call_and_return_conditional_losses_4628’
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
'__inference_conv2d_3_layer_call_fn_4646’
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
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4640’
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
2
4__inference_batch_normalization_6_layer_call_fn_4754
4__inference_batch_normalization_6_layer_call_fn_4691
4__inference_batch_normalization_6_layer_call_fn_4745
4__inference_batch_normalization_6_layer_call_fn_4700΄
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
ώ2ϋ
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4736
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4682
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4664
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4718΄
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
Π2Ν
&__inference_re_lu_6_layer_call_fn_4764’
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
A__inference_re_lu_6_layer_call_and_return_conditional_losses_4759’
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
2
1__inference_depthwise_conv2d_3_layer_call_fn_1462Ψ
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
annotationsͺ *8’5
30,???????????????????????????
¬2©
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1446Ψ
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
annotationsͺ *8’5
30,???????????????????????????
2
4__inference_batch_normalization_7_layer_call_fn_4872
4__inference_batch_normalization_7_layer_call_fn_4818
4__inference_batch_normalization_7_layer_call_fn_4863
4__inference_batch_normalization_7_layer_call_fn_4809΄
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
ώ2ϋ
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4800
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4836
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4854
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4782΄
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
Π2Ν
&__inference_re_lu_7_layer_call_fn_4882’
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
A__inference_re_lu_7_layer_call_and_return_conditional_losses_4877’
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
'__inference_conv2d_4_layer_call_fn_4895’
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
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4889’
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
2
4__inference_batch_normalization_8_layer_call_fn_5003
4__inference_batch_normalization_8_layer_call_fn_4949
4__inference_batch_normalization_8_layer_call_fn_4940
4__inference_batch_normalization_8_layer_call_fn_4994΄
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
ώ2ϋ
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4967
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4985
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4913
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4931΄
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
Π2Ν
&__inference_re_lu_8_layer_call_fn_5013’
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
A__inference_re_lu_8_layer_call_and_return_conditional_losses_5008’
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
2
7__inference_global_average_pooling2d_layer_call_fn_1665ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
Ί2·
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1653ΰ
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
annotationsͺ *@’=
;84????????????????????????????????????
2
&__inference_dropout_layer_call_fn_5040
&__inference_dropout_layer_call_fn_5035΄
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
A__inference_dropout_layer_call_and_return_conditional_losses_5030
A__inference_dropout_layer_call_and_return_conditional_losses_5025΄
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
$__inference_dense_layer_call_fn_5057’
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
?__inference_dense_layer_call_and_return_conditional_losses_5050’
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
1B/
"__inference_signature_wrapper_2913input_1Ζ
__inference__wrapped_model_714£E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ4’1
*’'
%"
input_1?????????1(
ͺ "$ͺ!

dense
dense³
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4035`BCDE2’/
(’%

inputs 
p
ͺ "$’!

0 
 ³
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4053`BCDE2’/
(’%

inputs 
p 
ͺ "$’!

0 
 κ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4089BCDEM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 κ
O__inference_batch_normalization_1_layer_call_and_return_conditional_losses_4107BCDEM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 
4__inference_batch_normalization_1_layer_call_fn_4062SBCDE2’/
(’%

inputs 
p
ͺ " 
4__inference_batch_normalization_1_layer_call_fn_4071SBCDE2’/
(’%

inputs 
p 
ͺ " Β
4__inference_batch_normalization_1_layer_call_fn_4116BCDEM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? Β
4__inference_batch_normalization_1_layer_call_fn_4125BCDEM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? ³
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4166`TUVW2’/
(’%

inputs 
p
ͺ "$’!

0 
 ³
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4184`TUVW2’/
(’%

inputs 
p 
ͺ "$’!

0 
 κ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4220TUVWM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 κ
O__inference_batch_normalization_2_layer_call_and_return_conditional_losses_4238TUVWM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 
4__inference_batch_normalization_2_layer_call_fn_4193STUVW2’/
(’%

inputs 
p
ͺ " 
4__inference_batch_normalization_2_layer_call_fn_4202STUVW2’/
(’%

inputs 
p 
ͺ " Β
4__inference_batch_normalization_2_layer_call_fn_4247TUVWM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? Β
4__inference_batch_normalization_2_layer_call_fn_4256TUVWM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? κ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4284fghiM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 κ
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4302fghiM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 ³
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4338`fghi2’/
(’%

inputs 
p
ͺ "$’!

0 
 ³
O__inference_batch_normalization_3_layer_call_and_return_conditional_losses_4356`fghi2’/
(’%

inputs 
p 
ͺ "$’!

0 
 Β
4__inference_batch_normalization_3_layer_call_fn_4311fghiM’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? Β
4__inference_batch_normalization_3_layer_call_fn_4320fghiM’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? 
4__inference_batch_normalization_3_layer_call_fn_4365Sfghi2’/
(’%

inputs 
p
ͺ " 
4__inference_batch_normalization_3_layer_call_fn_4374Sfghi2’/
(’%

inputs 
p 
ͺ " ³
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4415`xyz{2’/
(’%

inputs@
p
ͺ "$’!

0@
 ³
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4433`xyz{2’/
(’%

inputs@
p 
ͺ "$’!

0@
 κ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4469xyz{M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 κ
O__inference_batch_normalization_4_layer_call_and_return_conditional_losses_4487xyz{M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 
4__inference_batch_normalization_4_layer_call_fn_4442Sxyz{2’/
(’%

inputs@
p
ͺ "@
4__inference_batch_normalization_4_layer_call_fn_4451Sxyz{2’/
(’%

inputs@
p 
ͺ "@Β
4__inference_batch_normalization_4_layer_call_fn_4496xyz{M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Β
4__inference_batch_normalization_4_layer_call_fn_4505xyz{M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@·
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4533d2’/
(’%

inputs@
p
ͺ "$’!

0@
 ·
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4551d2’/
(’%

inputs@
p 
ͺ "$’!

0@
 ξ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4587M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "?’<
52
0+???????????????????????????@
 ξ
O__inference_batch_normalization_5_layer_call_and_return_conditional_losses_4605M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "?’<
52
0+???????????????????????????@
 
4__inference_batch_normalization_5_layer_call_fn_4560W2’/
(’%

inputs@
p
ͺ "@
4__inference_batch_normalization_5_layer_call_fn_4569W2’/
(’%

inputs@
p 
ͺ "@Ζ
4__inference_batch_normalization_5_layer_call_fn_4614M’J
C’@
:7
inputs+???????????????????????????@
p
ͺ "2/+???????????????????????????@Ζ
4__inference_batch_normalization_5_layer_call_fn_4623M’J
C’@
:7
inputs+???????????????????????????@
p 
ͺ "2/+???????????????????????????@π
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4664N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 π
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4682N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 Ή
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4718f3’0
)’&
 
inputs
p
ͺ "%’"

0
 Ή
O__inference_batch_normalization_6_layer_call_and_return_conditional_losses_4736f3’0
)’&
 
inputs
p 
ͺ "%’"

0
 Θ
4__inference_batch_normalization_6_layer_call_fn_4691N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Θ
4__inference_batch_normalization_6_layer_call_fn_4700N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????
4__inference_batch_normalization_6_layer_call_fn_4745Y3’0
)’&
 
inputs
p
ͺ "
4__inference_batch_normalization_6_layer_call_fn_4754Y3’0
)’&
 
inputs
p 
ͺ "Ή
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4782f?―°±3’0
)’&
 
inputs
p
ͺ "%’"

0
 Ή
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4800f?―°±3’0
)’&
 
inputs
p 
ͺ "%’"

0
 π
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4836?―°±N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 π
O__inference_batch_normalization_7_layer_call_and_return_conditional_losses_4854?―°±N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 
4__inference_batch_normalization_7_layer_call_fn_4809Y?―°±3’0
)’&
 
inputs
p
ͺ "
4__inference_batch_normalization_7_layer_call_fn_4818Y?―°±3’0
)’&
 
inputs
p 
ͺ "Θ
4__inference_batch_normalization_7_layer_call_fn_4863?―°±N’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Θ
4__inference_batch_normalization_7_layer_call_fn_4872?―°±N’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????Ή
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4913fΐΑΒΓ3’0
)’&
 
inputs
p
ͺ "%’"

0
 Ή
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4931fΐΑΒΓ3’0
)’&
 
inputs
p 
ͺ "%’"

0
 π
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4967ΐΑΒΓN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "@’=
63
0,???????????????????????????
 π
O__inference_batch_normalization_8_layer_call_and_return_conditional_losses_4985ΐΑΒΓN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "@’=
63
0,???????????????????????????
 
4__inference_batch_normalization_8_layer_call_fn_4940YΐΑΒΓ3’0
)’&
 
inputs
p
ͺ "
4__inference_batch_normalization_8_layer_call_fn_4949YΐΑΒΓ3’0
)’&
 
inputs
p 
ͺ "Θ
4__inference_batch_normalization_8_layer_call_fn_4994ΐΑΒΓN’K
D’A
;8
inputs,???????????????????????????
p
ͺ "30,???????????????????????????Θ
4__inference_batch_normalization_8_layer_call_fn_5003ΐΑΒΓN’K
D’A
;8
inputs,???????????????????????????
p 
ͺ "30,???????????????????????????θ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_39170123M’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "?’<
52
0+??????????????????????????? 
 θ
M__inference_batch_normalization_layer_call_and_return_conditional_losses_39350123M’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "?’<
52
0+??????????????????????????? 
 ±
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3971`01232’/
(’%

inputs 
p
ͺ "$’!

0 
 ±
M__inference_batch_normalization_layer_call_and_return_conditional_losses_3989`01232’/
(’%

inputs 
p 
ͺ "$’!

0 
 ΐ
2__inference_batch_normalization_layer_call_fn_39440123M’J
C’@
:7
inputs+??????????????????????????? 
p
ͺ "2/+??????????????????????????? ΐ
2__inference_batch_normalization_layer_call_fn_39530123M’J
C’@
:7
inputs+??????????????????????????? 
p 
ͺ "2/+??????????????????????????? 
2__inference_batch_normalization_layer_call_fn_3998S01232’/
(’%

inputs 
p
ͺ " 
2__inference_batch_normalization_layer_call_fn_4007S01232’/
(’%

inputs 
p 
ͺ " 
B__inference_conv2d_1_layer_call_and_return_conditional_losses_4142YN.’+
$’!

inputs 
ͺ "$’!

0 
 w
'__inference_conv2d_1_layer_call_fn_4148LN.’+
$’!

inputs 
ͺ " 
B__inference_conv2d_2_layer_call_and_return_conditional_losses_4391Yr.’+
$’!

inputs 
ͺ "$’!

0@
 w
'__inference_conv2d_2_layer_call_fn_4397Lr.’+
$’!

inputs 
ͺ "@‘
B__inference_conv2d_3_layer_call_and_return_conditional_losses_4640[.’+
$’!

inputs@
ͺ "%’"

0
 y
'__inference_conv2d_3_layer_call_fn_4646N.’+
$’!

inputs@
ͺ "’
B__inference_conv2d_4_layer_call_and_return_conditional_losses_4889\Ί/’,
%’"
 
inputs
ͺ "%’"

0
 z
'__inference_conv2d_4_layer_call_fn_4895OΊ/’,
%’"
 
inputs
ͺ "
@__inference_conv2d_layer_call_and_return_conditional_losses_3893Y*.’+
$’!

inputs1(
ͺ "$’!

0 
 u
%__inference_conv2d_layer_call_fn_3899L*.’+
$’!

inputs1(
ͺ " 
?__inference_dense_layer_call_and_return_conditional_losses_5050MΤΥ'’$
’

inputs	
ͺ "’

0
 h
$__inference_dense_layer_call_fn_5057@ΤΥ'’$
’

inputs	
ͺ "ΰ
L__inference_depthwise_conv2d_1_layer_call_and_return_conditional_losses_1026`I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+??????????????????????????? 
 Έ
1__inference_depthwise_conv2d_1_layer_call_fn_1042`I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+??????????????????????????? α
L__inference_depthwise_conv2d_2_layer_call_and_return_conditional_losses_1236I’F
?’<
:7
inputs+???????????????????????????@
ͺ "?’<
52
0+???????????????????????????@
 Ή
1__inference_depthwise_conv2d_2_layer_call_fn_1252I’F
?’<
:7
inputs+???????????????????????????@
ͺ "2/+???????????????????????????@γ
L__inference_depthwise_conv2d_3_layer_call_and_return_conditional_losses_1446¨J’G
@’=
;8
inputs,???????????????????????????
ͺ "@’=
63
0,???????????????????????????
 »
1__inference_depthwise_conv2d_3_layer_call_fn_1462¨J’G
@’=
;8
inputs,???????????????????????????
ͺ "30,???????????????????????????έ
I__inference_depthwise_conv2d_layer_call_and_return_conditional_losses_816<I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "?’<
52
0+??????????????????????????? 
 ΅
.__inference_depthwise_conv2d_layer_call_fn_832<I’F
?’<
:7
inputs+??????????????????????????? 
ͺ "2/+??????????????????????????? 
A__inference_dropout_layer_call_and_return_conditional_losses_5025L+’(
!’

inputs	
p
ͺ "’

0	
 
A__inference_dropout_layer_call_and_return_conditional_losses_5030L+’(
!’

inputs	
p 
ͺ "’

0	
 i
&__inference_dropout_layer_call_fn_5035?+’(
!’

inputs	
p
ͺ "	i
&__inference_dropout_layer_call_fn_5040?+’(
!’

inputs	
p 
ͺ "	ξ
F__inference_functional_1_layer_call_and_return_conditional_losses_3105£E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ<’9
2’/
%"
input_1?????????1(
p

 
ͺ "’

0
 ξ
F__inference_functional_1_layer_call_and_return_conditional_losses_3290£E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ<’9
2’/
%"
input_1?????????1(
p 

 
ͺ "’

0
 ν
F__inference_functional_1_layer_call_and_return_conditional_losses_3586’E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ;’8
1’.
$!
inputs?????????1(
p

 
ͺ "’

0
 ν
F__inference_functional_1_layer_call_and_return_conditional_losses_3771’E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ;’8
1’.
$!
inputs?????????1(
p 

 
ͺ "’

0
 Ζ
+__inference_functional_1_layer_call_fn_3342E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ<’9
2’/
%"
input_1?????????1(
p

 
ͺ "Ζ
+__inference_functional_1_layer_call_fn_3394E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ<’9
2’/
%"
input_1?????????1(
p 

 
ͺ "Ε
+__inference_functional_1_layer_call_fn_3823E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ;’8
1’.
$!
inputs?????????1(
p

 
ͺ "Ε
+__inference_functional_1_layer_call_fn_3875E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ;’8
1’.
$!
inputs?????????1(
p 

 
ͺ "Ϋ
R__inference_global_average_pooling2d_layer_call_and_return_conditional_losses_1653R’O
H’E
C@
inputs4????????????????????????????????????
ͺ ".’+
$!
0??????????????????
 ²
7__inference_global_average_pooling2d_layer_call_fn_1665wR’O
H’E
C@
inputs4????????????????????????????????????
ͺ "!??????????????????
A__inference_re_lu_1_layer_call_and_return_conditional_losses_4130V.’+
$’!

inputs 
ͺ "$’!

0 
 s
&__inference_re_lu_1_layer_call_fn_4135I.’+
$’!

inputs 
ͺ " 
A__inference_re_lu_2_layer_call_and_return_conditional_losses_4261V.’+
$’!

inputs 
ͺ "$’!

0 
 s
&__inference_re_lu_2_layer_call_fn_4266I.’+
$’!

inputs 
ͺ " 
A__inference_re_lu_3_layer_call_and_return_conditional_losses_4379V.’+
$’!

inputs 
ͺ "$’!

0 
 s
&__inference_re_lu_3_layer_call_fn_4384I.’+
$’!

inputs 
ͺ " 
A__inference_re_lu_4_layer_call_and_return_conditional_losses_4510V.’+
$’!

inputs@
ͺ "$’!

0@
 s
&__inference_re_lu_4_layer_call_fn_4515I.’+
$’!

inputs@
ͺ "@
A__inference_re_lu_5_layer_call_and_return_conditional_losses_4628V.’+
$’!

inputs@
ͺ "$’!

0@
 s
&__inference_re_lu_5_layer_call_fn_4633I.’+
$’!

inputs@
ͺ "@
A__inference_re_lu_6_layer_call_and_return_conditional_losses_4759X/’,
%’"
 
inputs
ͺ "%’"

0
 u
&__inference_re_lu_6_layer_call_fn_4764K/’,
%’"
 
inputs
ͺ "
A__inference_re_lu_7_layer_call_and_return_conditional_losses_4877X/’,
%’"
 
inputs
ͺ "%’"

0
 u
&__inference_re_lu_7_layer_call_fn_4882K/’,
%’"
 
inputs
ͺ "
A__inference_re_lu_8_layer_call_and_return_conditional_losses_5008X/’,
%’"
 
inputs
ͺ "%’"

0
 u
&__inference_re_lu_8_layer_call_fn_5013K/’,
%’"
 
inputs
ͺ "
?__inference_re_lu_layer_call_and_return_conditional_losses_4012V.’+
$’!

inputs 
ͺ "$’!

0 
 q
$__inference_re_lu_layer_call_fn_4017I.’+
$’!

inputs 
ͺ " Μ
"__inference_signature_wrapper_2913₯E*0123<BCDENTUVW`fghirxyz{¨?―°±ΊΐΑΒΓΤΥ6’3
’ 
,ͺ)
'
input_1
input_11("$ͺ!

dense
dense¦
P__inference_tf_op_layer_ExpandDims_layer_call_and_return_conditional_losses_3881R*’'
 ’

inputs1(
ͺ "$’!

01(
 ~
5__inference_tf_op_layer_ExpandDims_layer_call_fn_3886E*’'
 ’

inputs1(
ͺ "1(