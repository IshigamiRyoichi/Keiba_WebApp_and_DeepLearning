??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
rank_net/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namerank_net/dense_2/kernel
?
+rank_net/dense_2/kernel/Read/ReadVariableOpReadVariableOprank_net/dense_2/kernel*
_output_shapes

:*
dtype0
?
rank_net/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namerank_net/dense_2/bias
{
)rank_net/dense_2/bias/Read/ReadVariableOpReadVariableOprank_net/dense_2/bias*
_output_shapes
:*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
_output_shapes
: *
dtype0
?
rank_net/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_namerank_net/dense/kernel

)rank_net/dense/kernel/Read/ReadVariableOpReadVariableOprank_net/dense/kernel*
_output_shapes

:*
dtype0
~
rank_net/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namerank_net/dense/bias
w
'rank_net/dense/bias/Read/ReadVariableOpReadVariableOprank_net/dense/bias*
_output_shapes
:*
dtype0
?
rank_net/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*(
shared_namerank_net/dense_1/kernel
?
+rank_net/dense_1/kernel/Read/ReadVariableOpReadVariableOprank_net/dense_1/kernel*
_output_shapes

:*
dtype0
?
rank_net/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namerank_net/dense_1/bias
{
)rank_net/dense_1/bias/Read/ReadVariableOpReadVariableOprank_net/dense_1/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	dense
o
oi_minus_oj
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures


0
1
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
6
iter
	decay
learning_rate
momentum
 
*
0
1
2
3
4
5
*
0
1
2
3
4
5
?
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
	variables
!metrics

"layers
 
h

kernel
bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
h

kernel
bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
PN
VARIABLE_VALUErank_net/dense_2/kernel#o/kernel/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUErank_net/dense_2/bias!o/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
trainable_variables
+layer_metrics
,non_trainable_variables
-layer_regularization_losses
	variables
.metrics

/layers
 
 
 
?
regularization_losses
trainable_variables
0layer_metrics
1non_trainable_variables
2layer_regularization_losses
	variables
3metrics

4layers
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErank_net/dense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUErank_net/dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUErank_net/dense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUErank_net/dense_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

50
61


0
1
2
3
 

0
1

0
1
?
#regularization_losses
$trainable_variables
7layer_metrics
8non_trainable_variables
9layer_regularization_losses
%	variables
:metrics

;layers
 

0
1

0
1
?
'regularization_losses
(trainable_variables
<layer_metrics
=non_trainable_variables
>layer_regularization_losses
)	variables
?metrics

@layers
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
4
	Atotal
	Bcount
C	variables
D	keras_api
D
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

A0
B1

C	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

H	variables
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2rank_net/dense/kernelrank_net/dense/biasrank_net/dense_1/kernelrank_net/dense_1/biasrank_net/dense_2/kernelrank_net/dense_2/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_50208
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+rank_net/dense_2/kernel/Read/ReadVariableOp)rank_net/dense_2/bias/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOp)rank_net/dense/kernel/Read/ReadVariableOp'rank_net/dense/bias/Read/ReadVariableOp+rank_net/dense_1/kernel/Read/ReadVariableOp)rank_net/dense_1/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_50404
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamerank_net/dense_2/kernelrank_net/dense_2/biasSGD/iter	SGD/decaySGD/learning_rateSGD/momentumrank_net/dense/kernelrank_net/dense/biasrank_net/dense_1/kernelrank_net/dense_1/biastotalcounttotal_1count_1*
Tin
2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_50456??
?$
?
C__inference_rank_net_layer_call_and_return_conditional_losses_50184
input_1
input_2
dense_50157:
dense_50159:
dense_1_50165:
dense_1_50167:
dense_2_50173:
dense_2_50175:
identity??dense/StatefulPartitionedCall?dense/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?!dense_1/StatefulPartitionedCall_1?dense_2/StatefulPartitionedCall?!dense_2/StatefulPartitionedCall_1?
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_50157dense_50159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_500252
dense/StatefulPartitionedCall?
dense/StatefulPartitionedCall_1StatefulPartitionedCallinput_2dense_50157dense_50159*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_500252!
dense/StatefulPartitionedCall_1?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_50165dense_1_50167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_500452!
dense_1/StatefulPartitionedCall?
!dense_1/StatefulPartitionedCall_1StatefulPartitionedCall(dense/StatefulPartitionedCall_1:output:0dense_1_50165dense_1_50167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_500452#
!dense_1/StatefulPartitionedCall_1?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_50173dense_2_50175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_500642!
dense_2/StatefulPartitionedCall?
!dense_2/StatefulPartitionedCall_1StatefulPartitionedCall*dense_1/StatefulPartitionedCall_1:output:0dense_2_50173dense_2_50175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_500642#
!dense_2/StatefulPartitionedCall_1?
subtract/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*dense_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_500792
subtract/PartitionedCall?
activation/SigmoidSigmoid!subtract/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
activation/Sigmoidq
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall"^dense_1/StatefulPartitionedCall_1 ^dense_2/StatefulPartitionedCall"^dense_2/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense/StatefulPartitionedCall_1dense/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dense_1/StatefulPartitionedCall_1!dense_1/StatefulPartitionedCall_12B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dense_2/StatefulPartitionedCall_1!dense_2/StatefulPartitionedCall_1:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?

?
(__inference_rank_net_layer_call_fn_50098
input_1
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_rank_net_layer_call_and_return_conditional_losses_500832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?8
?
C__inference_rank_net_layer_call_and_return_conditional_losses_50249
inputs_0
inputs_16
$dense_matmul_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:8
&dense_1_matmul_readvariableop_resource:5
'dense_1_biasadd_readvariableop_resource:8
&dense_2_matmul_readvariableop_resource:5
'dense_2_biasadd_readvariableop_resource:
identity??dense/BiasAdd/ReadVariableOp?dense/BiasAdd_1/ReadVariableOp?dense/MatMul/ReadVariableOp?dense/MatMul_1/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp? dense_1/BiasAdd_1/ReadVariableOp?dense_1/MatMul/ReadVariableOp?dense_1/MatMul_1/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/BiasAdd_1/ReadVariableOp?dense_2/MatMul/ReadVariableOp?dense_2/MatMul_1/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs_0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAddp
dense/LeakyRelu	LeakyReludense/BiasAdd:output:0*'
_output_shapes
:?????????2
dense/LeakyRelu?
dense/MatMul_1/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense/MatMul_1/ReadVariableOp?
dense/MatMul_1MatMulinputs_1%dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul_1?
dense/BiasAdd_1/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense/BiasAdd_1/ReadVariableOp?
dense/BiasAdd_1BiasAdddense/MatMul_1:product:0&dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd_1v
dense/LeakyRelu_1	LeakyReludense/BiasAdd_1:output:0*'
_output_shapes
:?????????2
dense/LeakyRelu_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMuldense/LeakyRelu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddv
dense_1/LeakyRelu	LeakyReludense_1/BiasAdd:output:0*'
_output_shapes
:?????????2
dense_1/LeakyRelu?
dense_1/MatMul_1/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_1/MatMul_1/ReadVariableOp?
dense_1/MatMul_1MatMuldense/LeakyRelu_1:activations:0'dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul_1?
 dense_1/BiasAdd_1/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_1/BiasAdd_1/ReadVariableOp?
dense_1/BiasAdd_1BiasAdddense_1/MatMul_1:product:0(dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAdd_1|
dense_1/LeakyRelu_1	LeakyReludense_1/BiasAdd_1:output:0*'
_output_shapes
:?????????2
dense_1/LeakyRelu_1?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMuldense_1/LeakyRelu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd?
dense_2/MatMul_1/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02!
dense_2/MatMul_1/ReadVariableOp?
dense_2/MatMul_1MatMul!dense_1/LeakyRelu_1:activations:0'dense_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/MatMul_1?
 dense_2/BiasAdd_1/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 dense_2/BiasAdd_1/ReadVariableOp?
dense_2/BiasAdd_1BiasAdddense_2/MatMul_1:product:0(dense_2/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_2/BiasAdd_1?
subtract/subSubdense_2/BiasAdd:output:0dense_2/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
subtract/subw
activation/SigmoidSigmoidsubtract/sub:z:0*
T0*'
_output_shapes
:?????????2
activation/Sigmoidq
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/BiasAdd_1/ReadVariableOp^dense/MatMul/ReadVariableOp^dense/MatMul_1/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp!^dense_1/BiasAdd_1/ReadVariableOp^dense_1/MatMul/ReadVariableOp ^dense_1/MatMul_1/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/BiasAdd_1/ReadVariableOp^dense_2/MatMul/ReadVariableOp ^dense_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/BiasAdd_1/ReadVariableOpdense/BiasAdd_1/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2>
dense/MatMul_1/ReadVariableOpdense/MatMul_1/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2D
 dense_1/BiasAdd_1/ReadVariableOp dense_1/BiasAdd_1/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2B
dense_1/MatMul_1/ReadVariableOpdense_1/MatMul_1/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/BiasAdd_1/ReadVariableOp dense_2/BiasAdd_1/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp2B
dense_2/MatMul_1/ReadVariableOpdense_2/MatMul_1/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
(__inference_rank_net_layer_call_fn_50267
inputs_0
inputs_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_rank_net_layer_call_and_return_conditional_losses_500832
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?C
?
 __inference__wrapped_model_50005
input_1
input_2?
-rank_net_dense_matmul_readvariableop_resource:<
.rank_net_dense_biasadd_readvariableop_resource:A
/rank_net_dense_1_matmul_readvariableop_resource:>
0rank_net_dense_1_biasadd_readvariableop_resource:A
/rank_net_dense_2_matmul_readvariableop_resource:>
0rank_net_dense_2_biasadd_readvariableop_resource:
identity??%rank_net/dense/BiasAdd/ReadVariableOp?'rank_net/dense/BiasAdd_1/ReadVariableOp?$rank_net/dense/MatMul/ReadVariableOp?&rank_net/dense/MatMul_1/ReadVariableOp?'rank_net/dense_1/BiasAdd/ReadVariableOp?)rank_net/dense_1/BiasAdd_1/ReadVariableOp?&rank_net/dense_1/MatMul/ReadVariableOp?(rank_net/dense_1/MatMul_1/ReadVariableOp?'rank_net/dense_2/BiasAdd/ReadVariableOp?)rank_net/dense_2/BiasAdd_1/ReadVariableOp?&rank_net/dense_2/MatMul/ReadVariableOp?(rank_net/dense_2/MatMul_1/ReadVariableOp?
$rank_net/dense/MatMul/ReadVariableOpReadVariableOp-rank_net_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02&
$rank_net/dense/MatMul/ReadVariableOp?
rank_net/dense/MatMulMatMulinput_1,rank_net/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/MatMul?
%rank_net/dense/BiasAdd/ReadVariableOpReadVariableOp.rank_net_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%rank_net/dense/BiasAdd/ReadVariableOp?
rank_net/dense/BiasAddBiasAddrank_net/dense/MatMul:product:0-rank_net/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/BiasAdd?
rank_net/dense/LeakyRelu	LeakyRelurank_net/dense/BiasAdd:output:0*'
_output_shapes
:?????????2
rank_net/dense/LeakyRelu?
&rank_net/dense/MatMul_1/ReadVariableOpReadVariableOp-rank_net_dense_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&rank_net/dense/MatMul_1/ReadVariableOp?
rank_net/dense/MatMul_1MatMulinput_2.rank_net/dense/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/MatMul_1?
'rank_net/dense/BiasAdd_1/ReadVariableOpReadVariableOp.rank_net_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rank_net/dense/BiasAdd_1/ReadVariableOp?
rank_net/dense/BiasAdd_1BiasAdd!rank_net/dense/MatMul_1:product:0/rank_net/dense/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense/BiasAdd_1?
rank_net/dense/LeakyRelu_1	LeakyRelu!rank_net/dense/BiasAdd_1:output:0*'
_output_shapes
:?????????2
rank_net/dense/LeakyRelu_1?
&rank_net/dense_1/MatMul/ReadVariableOpReadVariableOp/rank_net_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&rank_net/dense_1/MatMul/ReadVariableOp?
rank_net/dense_1/MatMulMatMul&rank_net/dense/LeakyRelu:activations:0.rank_net/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/MatMul?
'rank_net/dense_1/BiasAdd/ReadVariableOpReadVariableOp0rank_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rank_net/dense_1/BiasAdd/ReadVariableOp?
rank_net/dense_1/BiasAddBiasAdd!rank_net/dense_1/MatMul:product:0/rank_net/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/BiasAdd?
rank_net/dense_1/LeakyRelu	LeakyRelu!rank_net/dense_1/BiasAdd:output:0*'
_output_shapes
:?????????2
rank_net/dense_1/LeakyRelu?
(rank_net/dense_1/MatMul_1/ReadVariableOpReadVariableOp/rank_net_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net/dense_1/MatMul_1/ReadVariableOp?
rank_net/dense_1/MatMul_1MatMul(rank_net/dense/LeakyRelu_1:activations:00rank_net/dense_1/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/MatMul_1?
)rank_net/dense_1/BiasAdd_1/ReadVariableOpReadVariableOp0rank_net_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net/dense_1/BiasAdd_1/ReadVariableOp?
rank_net/dense_1/BiasAdd_1BiasAdd#rank_net/dense_1/MatMul_1:product:01rank_net/dense_1/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_1/BiasAdd_1?
rank_net/dense_1/LeakyRelu_1	LeakyRelu#rank_net/dense_1/BiasAdd_1:output:0*'
_output_shapes
:?????????2
rank_net/dense_1/LeakyRelu_1?
&rank_net/dense_2/MatMul/ReadVariableOpReadVariableOp/rank_net_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02(
&rank_net/dense_2/MatMul/ReadVariableOp?
rank_net/dense_2/MatMulMatMul(rank_net/dense_1/LeakyRelu:activations:0.rank_net/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/MatMul?
'rank_net/dense_2/BiasAdd/ReadVariableOpReadVariableOp0rank_net_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'rank_net/dense_2/BiasAdd/ReadVariableOp?
rank_net/dense_2/BiasAddBiasAdd!rank_net/dense_2/MatMul:product:0/rank_net/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/BiasAdd?
(rank_net/dense_2/MatMul_1/ReadVariableOpReadVariableOp/rank_net_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(rank_net/dense_2/MatMul_1/ReadVariableOp?
rank_net/dense_2/MatMul_1MatMul*rank_net/dense_1/LeakyRelu_1:activations:00rank_net/dense_2/MatMul_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/MatMul_1?
)rank_net/dense_2/BiasAdd_1/ReadVariableOpReadVariableOp0rank_net_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)rank_net/dense_2/BiasAdd_1/ReadVariableOp?
rank_net/dense_2/BiasAdd_1BiasAdd#rank_net/dense_2/MatMul_1:product:01rank_net/dense_2/BiasAdd_1/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
rank_net/dense_2/BiasAdd_1?
rank_net/subtract/subSub!rank_net/dense_2/BiasAdd:output:0#rank_net/dense_2/BiasAdd_1:output:0*
T0*'
_output_shapes
:?????????2
rank_net/subtract/sub?
rank_net/activation/SigmoidSigmoidrank_net/subtract/sub:z:0*
T0*'
_output_shapes
:?????????2
rank_net/activation/Sigmoidz
IdentityIdentityrank_net/activation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp&^rank_net/dense/BiasAdd/ReadVariableOp(^rank_net/dense/BiasAdd_1/ReadVariableOp%^rank_net/dense/MatMul/ReadVariableOp'^rank_net/dense/MatMul_1/ReadVariableOp(^rank_net/dense_1/BiasAdd/ReadVariableOp*^rank_net/dense_1/BiasAdd_1/ReadVariableOp'^rank_net/dense_1/MatMul/ReadVariableOp)^rank_net/dense_1/MatMul_1/ReadVariableOp(^rank_net/dense_2/BiasAdd/ReadVariableOp*^rank_net/dense_2/BiasAdd_1/ReadVariableOp'^rank_net/dense_2/MatMul/ReadVariableOp)^rank_net/dense_2/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2N
%rank_net/dense/BiasAdd/ReadVariableOp%rank_net/dense/BiasAdd/ReadVariableOp2R
'rank_net/dense/BiasAdd_1/ReadVariableOp'rank_net/dense/BiasAdd_1/ReadVariableOp2L
$rank_net/dense/MatMul/ReadVariableOp$rank_net/dense/MatMul/ReadVariableOp2P
&rank_net/dense/MatMul_1/ReadVariableOp&rank_net/dense/MatMul_1/ReadVariableOp2R
'rank_net/dense_1/BiasAdd/ReadVariableOp'rank_net/dense_1/BiasAdd/ReadVariableOp2V
)rank_net/dense_1/BiasAdd_1/ReadVariableOp)rank_net/dense_1/BiasAdd_1/ReadVariableOp2P
&rank_net/dense_1/MatMul/ReadVariableOp&rank_net/dense_1/MatMul/ReadVariableOp2T
(rank_net/dense_1/MatMul_1/ReadVariableOp(rank_net/dense_1/MatMul_1/ReadVariableOp2R
'rank_net/dense_2/BiasAdd/ReadVariableOp'rank_net/dense_2/BiasAdd/ReadVariableOp2V
)rank_net/dense_2/BiasAdd_1/ReadVariableOp)rank_net/dense_2/BiasAdd_1/ReadVariableOp2P
&rank_net/dense_2/MatMul/ReadVariableOp&rank_net/dense_2/MatMul/ReadVariableOp2T
(rank_net/dense_2/MatMul_1/ReadVariableOp(rank_net/dense_2/MatMul_1/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
T
(__inference_subtract_layer_call_fn_50298
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_500792
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
o
C__inference_subtract_layer_call_and_return_conditional_losses_50292
inputs_0
inputs_1
identityW
subSubinputs_0inputs_1*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?=
?
!__inference__traced_restore_50456
file_prefix:
(assignvariableop_rank_net_dense_2_kernel:6
(assignvariableop_1_rank_net_dense_2_bias:%
assignvariableop_2_sgd_iter:	 &
assignvariableop_3_sgd_decay: .
$assignvariableop_4_sgd_learning_rate: )
assignvariableop_5_sgd_momentum: :
(assignvariableop_6_rank_net_dense_kernel:4
&assignvariableop_7_rank_net_dense_bias:<
*assignvariableop_8_rank_net_dense_1_kernel:6
(assignvariableop_9_rank_net_dense_1_bias:#
assignvariableop_10_total: #
assignvariableop_11_count: %
assignvariableop_12_total_1: %
assignvariableop_13_count_1: 
identity_15??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*P
_output_shapes>
<:::::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp(assignvariableop_rank_net_dense_2_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp(assignvariableop_1_rank_net_dense_2_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_sgd_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_sgd_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rank_net_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp&assignvariableop_7_rank_net_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_rank_net_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_rank_net_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_totalIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_countIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_total_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_139
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_14Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_14f
Identity_15IdentityIdentity_14:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_15?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_15Identity_15:output:0*1
_input_shapes 
: : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132(
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
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_50045

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_50318

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_500252
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_1_layer_call_fn_50338

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_500452
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_2_layer_call_fn_50286

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_500642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_50309

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_50277

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
__inference__traced_save_50404
file_prefix6
2savev2_rank_net_dense_2_kernel_read_readvariableop4
0savev2_rank_net_dense_2_bias_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop4
0savev2_rank_net_dense_kernel_read_readvariableop2
.savev2_rank_net_dense_bias_read_readvariableop6
2savev2_rank_net_dense_1_kernel_read_readvariableop4
0savev2_rank_net_dense_1_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
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
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
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
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B#o/kernel/.ATTRIBUTES/VARIABLE_VALUEB!o/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_rank_net_dense_2_kernel_read_readvariableop0savev2_rank_net_dense_2_bias_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop0savev2_rank_net_dense_kernel_read_readvariableop.savev2_rank_net_dense_bias_read_readvariableop2savev2_rank_net_dense_1_kernel_read_readvariableop0savev2_rank_net_dense_1_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*W
_input_shapesF
D: ::: : : : ::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$	 

_output_shapes

:: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
#__inference_signature_wrapper_50208
input_1
input_2
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_500052
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_2
?
?
B__inference_dense_1_layer_call_and_return_conditional_losses_50329

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_2_layer_call_and_return_conditional_losses_50064

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
@__inference_dense_layer_call_and_return_conditional_losses_50025

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd^
	LeakyRelu	LeakyReluBiasAdd:output:0*'
_output_shapes
:?????????2
	LeakyRelur
IdentityIdentityLeakyRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?$
?
C__inference_rank_net_layer_call_and_return_conditional_losses_50083

inputs
inputs_1
dense_50026:
dense_50028:
dense_1_50046:
dense_1_50048:
dense_2_50065:
dense_2_50067:
identity??dense/StatefulPartitionedCall?dense/StatefulPartitionedCall_1?dense_1/StatefulPartitionedCall?!dense_1/StatefulPartitionedCall_1?dense_2/StatefulPartitionedCall?!dense_2/StatefulPartitionedCall_1?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_50026dense_50028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_500252
dense/StatefulPartitionedCall?
dense/StatefulPartitionedCall_1StatefulPartitionedCallinputs_1dense_50026dense_50028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_500252!
dense/StatefulPartitionedCall_1?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_50046dense_1_50048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_500452!
dense_1/StatefulPartitionedCall?
!dense_1/StatefulPartitionedCall_1StatefulPartitionedCall(dense/StatefulPartitionedCall_1:output:0dense_1_50046dense_1_50048*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_500452#
!dense_1/StatefulPartitionedCall_1?
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_50065dense_2_50067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_500642!
dense_2/StatefulPartitionedCall?
!dense_2/StatefulPartitionedCall_1StatefulPartitionedCall*dense_1/StatefulPartitionedCall_1:output:0dense_2_50065dense_2_50067*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_500642#
!dense_2/StatefulPartitionedCall_1?
subtract/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*dense_2/StatefulPartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *L
fGRE
C__inference_subtract_layer_call_and_return_conditional_losses_500792
subtract/PartitionedCall?
activation/SigmoidSigmoid!subtract/PartitionedCall:output:0*
T0*'
_output_shapes
:?????????2
activation/Sigmoidq
IdentityIdentityactivation/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^dense/StatefulPartitionedCall ^dense/StatefulPartitionedCall_1 ^dense_1/StatefulPartitionedCall"^dense_1/StatefulPartitionedCall_1 ^dense_2/StatefulPartitionedCall"^dense_2/StatefulPartitionedCall_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:?????????:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense/StatefulPartitionedCall_1dense/StatefulPartitionedCall_12B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2F
!dense_1/StatefulPartitionedCall_1!dense_1/StatefulPartitionedCall_12B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2F
!dense_2/StatefulPartitionedCall_1!dense_2/StatefulPartitionedCall_1:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
m
C__inference_subtract_layer_call_and_return_conditional_losses_50079

inputs
inputs_1
identityU
subSubinputsinputs_1*
T0*'
_output_shapes
:?????????2
sub[
IdentityIdentitysub:z:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?P
?
	dense
o
oi_minus_oj
	optimizer
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
*J&call_and_return_all_conditional_losses
K__call__
L_default_save_signature"
_tf_keras_model
.

0
1"
trackable_list_wrapper
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*M&call_and_return_all_conditional_losses
N__call__"
_tf_keras_layer
?
regularization_losses
trainable_variables
	variables
	keras_api
*O&call_and_return_all_conditional_losses
P__call__"
_tf_keras_layer
I
iter
	decay
learning_rate
momentum"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
?
regularization_losses
trainable_variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
	variables
!metrics

"layers
K__call__
L_default_save_signature
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
,
Qserving_default"
signature_map
?

kernel
bias
#regularization_losses
$trainable_variables
%	variables
&	keras_api
*R&call_and_return_all_conditional_losses
S__call__"
_tf_keras_layer
?

kernel
bias
'regularization_losses
(trainable_variables
)	variables
*	keras_api
*T&call_and_return_all_conditional_losses
U__call__"
_tf_keras_layer
):'2rank_net/dense_2/kernel
#:!2rank_net/dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
trainable_variables
+layer_metrics
,non_trainable_variables
-layer_regularization_losses
	variables
.metrics

/layers
N__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
trainable_variables
0layer_metrics
1non_trainable_variables
2layer_regularization_losses
	variables
3metrics

4layers
P__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
':%2rank_net/dense/kernel
!:2rank_net/dense/bias
):'2rank_net/dense_1/kernel
#:!2rank_net/dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
<

0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
#regularization_losses
$trainable_variables
7layer_metrics
8non_trainable_variables
9layer_regularization_losses
%	variables
:metrics

;layers
S__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
'regularization_losses
(trainable_variables
<layer_metrics
=non_trainable_variables
>layer_regularization_losses
)	variables
?metrics

@layers
U__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
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
N
	Atotal
	Bcount
C	variables
D	keras_api"
_tf_keras_metric
^
	Etotal
	Fcount
G
_fn_kwargs
H	variables
I	keras_api"
_tf_keras_metric
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
:  (2total
:  (2count
.
A0
B1"
trackable_list_wrapper
-
C	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
E0
F1"
trackable_list_wrapper
-
H	variables"
_generic_user_object
?2?
C__inference_rank_net_layer_call_and_return_conditional_losses_50249
C__inference_rank_net_layer_call_and_return_conditional_losses_50184?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_rank_net_layer_call_fn_50098
(__inference_rank_net_layer_call_fn_50267?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference__wrapped_model_50005input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_2_layer_call_and_return_conditional_losses_50277?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_2_layer_call_fn_50286?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_subtract_layer_call_and_return_conditional_losses_50292?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
(__inference_subtract_layer_call_fn_50298?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_50208input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_50309?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_50318?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_1_layer_call_and_return_conditional_losses_50329?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_1_layer_call_fn_50338?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_50005?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "3?0
.
output_1"?
output_1??????????
B__inference_dense_1_layer_call_and_return_conditional_losses_50329\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_1_layer_call_fn_50338O/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_2_layer_call_and_return_conditional_losses_50277\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_2_layer_call_fn_50286O/?,
%?"
 ?
inputs?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_50309\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? x
%__inference_dense_layer_call_fn_50318O/?,
%?"
 ?
inputs?????????
? "???????????
C__inference_rank_net_layer_call_and_return_conditional_losses_50184?X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "%?"
?
0?????????
? ?
C__inference_rank_net_layer_call_and_return_conditional_losses_50249?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
(__inference_rank_net_layer_call_fn_50098|X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????
? "???????????
(__inference_rank_net_layer_call_fn_50267~Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "???????????
#__inference_signature_wrapper_50208?i?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????"3?0
.
output_1"?
output_1??????????
C__inference_subtract_layer_call_and_return_conditional_losses_50292?Z?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "%?"
?
0?????????
? ?
(__inference_subtract_layer_call_fn_50298vZ?W
P?M
K?H
"?
inputs/0?????????
"?
inputs/1?????????
? "??????????