
module network_layers
  use network_utils
  implicit none

!========================== abstract layer ==========================

  type, abstract :: layer
    integer :: inputs, outputs
    class(Activation), allocatable :: func
    real, allocatable :: signals(:,:)
  contains
    procedure(layer_pass_signals), deferred :: run
    procedure(layer_train_forward), deferred :: train_forward
    procedure(layer_train_backward), deferred :: train_backward
    procedure(layer_random_init), deferred :: random_init
    generic :: fixed_init => f_init1, f_init2
    procedure(layer_finit1), deferred :: f_init1
    procedure :: f_init2 => unimplemented_fixed_init2
  end type layer

  type, abstract, extends(layer) :: connectom
    real, allocatable :: weights(:,:), derivatives(:,:)
  contains
    procedure :: set_layout => set_weights_size
    procedure :: random_init => set_weights_random
    procedure :: f_init1 => single_value_init
    procedure :: f_init2 => complete_init
    procedure :: calc_grad => gradient_connectom
    procedure :: internal_signal_transport => connectom_signal_transport
    procedure :: train_backward => train_backward_connectom
  end type connectom

!========================== interface declaration ==========================

  abstract interface
    pure function layer_pass_signals(this,signals) result (out)
    import layer
      class(layer), intent(in) :: this
      real, intent(in) :: signals(:,:) ! 1st dim this%inputs long
      real :: out(this%outputs, size(signals,2))
    end function layer_pass_signals

    function layer_train_forward(this,signals) result (out)
    import layer
      class(layer), intent(inout) :: this
      real, intent(in) :: signals(:,:) ! 1st dim this%inputs long
      real :: out(this%outputs, size(signals,2))
    end function layer_train_forward

    function layer_train_backward(this,error, alpha) result (out)
      import layer
      class(layer), intent(inout) :: this
      real, intent(in) :: alpha, error(:,:) ! 1st dim this%outputs long
      real :: out(this%inputs, size(error,2))
    end function layer_train_backward

    subroutine layer_random_init(this)
      import layer
      class(layer), intent(inout) :: this
    end subroutine layer_random_init

    pure subroutine layer_finit1(this, value)
      import layer
      class(layer), intent(inout) :: this
      real, intent(in) :: value
    end subroutine layer_finit1

  end interface

!========================== useful definitions ==========================

  type, extends(connectom) :: bias_layer
  contains
    procedure :: set_layout => set_biased_layout
    procedure :: train_forward => train_forward_biased_layer
    procedure :: run => interpret_biased_layer
  end type bias_layer

  type, extends(connectom) :: linear_layer
  contains
    procedure :: set_layout => set_linear_layout
    procedure :: train_forward => train_forward_linear_layer
    procedure :: run => interpret_linear_layer
  end type linear_layer

  interface bias_layer
    module procedure bias_layer_from_activation
    module procedure bias_layer_from_activation_and_size
  end interface bias_layer

contains

  subroutine unimplemented_fixed_init2(this, values)
    class(layer), intent(inout) :: this
    real, intent(in) :: values(:,:)
    stop
  end subroutine unimplemented_fixed_init2

  pure subroutine single_value_init(this, value)
    class(connectom), intent(inout) :: this
    real, intent(in) :: value
    this%weights = value
  end subroutine single_value_init

  subroutine complete_init(this, values)
    class(connectom), intent(inout) :: this
    real, intent(in) :: values (:,:)
    this%weights = values
  end subroutine complete_init

  subroutine set_biased_layout(this, inputs, outputs)
    class(bias_layer), intent(inout) :: this
    integer, intent(in) :: inputs, outputs
    this%inputs = inputs
    this%outputs = outputs
    if (allocated(this%weights)) deallocate(this%weights)
    allocate(this%weights(outputs,inputs+1))
  end subroutine set_biased_layout

  subroutine set_linear_layout(this, inputs, outputs)
    class(linear_layer), intent(inout) :: this
    integer, intent(in) :: inputs, outputs
    this%inputs = inputs
    this%outputs = outputs
    if (allocated(this%weights)) deallocate(this%weights)
    allocate(this%weights(outputs,inputs))
  end subroutine set_linear_layout

  subroutine set_weights_size(this, inputs, outputs)
    class(connectom), intent(inout) :: this ! HAS to be inout, or other params are lost !!
    integer, intent(in) :: inputs, outputs
    this%inputs = inputs
    this%outputs = outputs
    if (allocated(this%weights)) deallocate(this%weights)
    allocate(this%weights(outputs,inputs))
  end subroutine set_weights_size

  function bias_layer_from_activation(func) result (out)
    class(Activation), intent(in) :: func
    type(bias_layer) :: out
    allocate(out%func, source=func)
  end function bias_layer_from_activation

  function bias_layer_from_activation_and_size(func, inputs, outputs) result (out)
    class(Activation), intent(in) :: func
    type(bias_layer) :: out
    integer, intent(in) :: inputs, outputs
    call out%set_layout(inputs, outputs)
    allocate(out%func, source=func)
  end function bias_layer_from_activation_and_size

  function bias_layer_from_size(inputs, outputs) result (out)
    type(bias_layer) :: out
    integer, intent(in) :: inputs, outputs
    call out%set_layout(inputs, outputs)
  end function bias_layer_from_size

  subroutine set_weights_random(this)
    class(connectom), intent(inout) :: this
    call random_number(this%weights)
  end subroutine set_weights_random

  pure function interpret_biased_layer(this, signals) result (out)
    class(bias_layer), intent(in) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs,size(signals,2)), wrapper(this%inputs+1,size(signals,2))
    wrapper(this%inputs+1,:)=1
    wrapper(: this%inputs,:)=signals
    out = matmul(this%weights, wrapper)
    if (allocated(this%func)) then
      out = this%func%activate(out)
    end if
  end function interpret_biased_layer

  pure function interpret_linear_layer(this, signals) result (out)
    class(linear_layer), intent(in) :: this
    real, intent(in) :: signals(:,:)
    real :: out(this%outputs,size(signals,2))
    out = matmul(this%weights, signals)
    if (allocated(this%func)) then
      out = this%func%activate(out)
    end if
  end function interpret_linear_layer

  function connectom_signal_transport(this) result(out)
    class(connectom), intent(inout)::this
    real :: out(this%outputs, size(this%signals,2))
    out = matmul(   &
      this%weights, &
      this%signals  &
    )

    if (allocated(this%func)) then
      if (allocated(this%derivatives)) deallocate(this%derivatives)
      allocate(this%derivatives, mold=out)!source = this%func%derivative(out))
      block
        integer :: i
        do concurrent(i=1:size(this%signals,2))
          this%derivatives(:,i)=this%func%derivative(out(:,i))
          out(:,i) = this%func%activate(out(:,i))
        end do
      end block
    end if
  end function connectom_signal_transport

  function train_forward_biased_layer(this, signals) result (out)
    class(bias_layer), intent(inout) :: this
    real, intent(in) :: signals(:, :) ! 1st dim needs to be this%inputs long!
    real :: out(this%outputs,size(signals,2))

    if (allocated(this%signals)) deallocate(this%signals)
    allocate(this%signals(this%inputs+1,size(signals,2)))
    this%signals(this%inputs+1, :)=1
    this%signals(: this%inputs, :)=signals

    out = this%internal_signal_transport()
  end function train_forward_biased_layer

  function train_forward_linear_layer(this, signals) result (out)
    class(linear_layer), intent(inout) :: this
    real, intent(in) :: signals(:, :) ! 1st dim needs to be this%inputs long!
    real :: out(this%outputs,size(signals,2))

    if (allocated(this%signals)) deallocate(this%signals)
    allocate(this%signals, source=signals)

    out = this%internal_signal_transport()
  end function train_forward_linear_layer

  function gradient_connectom(this, error, corrections) result (prev_error)
    class(connectom), intent(in) :: this
    real, intent(in) :: error(:,:) ! 1st dim needs to be this%outputs long!
    real :: prev_error(this%inputs, size(error,2))
    real, intent(out) :: corrections(size(this%weights,1), size(this%weights,2))

    if (.not. allocated(this%signals) .or. (.not. allocated(this%derivatives) .and. allocated(this%func))) then
      print *, "THIS NEEDS SIGNALS AND DERIVATIVES! RUN TRAIN_FORWARD FIRST!!!"
      stop
    end if

    block
      real :: derivative_times_error(size(error,1), size(error,2))
      if(allocated(this%func)) then
        derivative_times_error = this%derivatives * error
      else
        derivative_times_error = error
      end if

      prev_error = matmul(                         &
        transpose(                                 &
          this%weights(:,1:size(this%weights,2)-1) &
        ),                                         &
        derivative_times_error                     &
      )
    end block

    corrections = matmul( &
      error,              &
      transpose(          &
        this%signals      &
      )                   &
    ) / size(error,2)

  end function gradient_connectom

  function train_backward_connectom(this, error, alpha) result (prev_error)
    class(connectom), intent(inout)::this
    real,intent(in) :: error(:,:), alpha
    real :: prev_error(this%inputs, size(error,2))!,corrections(this%outputs, this%inputs+1)

    prev_error = basic_SGD(this,error,alpha)
    deallocate(this%signals)
    deallocate(this%derivatives)
  end function train_backward_connectom

  function basic_SGD(this, error, alpha) result (prev_error)
    class(connectom), intent(inout)::this
    real, intent(in) :: error(:,:), alpha ! error needs to be this%outputs long in 1st dim
    real :: corrections(size(this%weights,1),size(this%weights,2)), prev_error(this%inputs, size(error,2))

    prev_error=this%calc_grad(error, corrections)
    this%weights = this%weights - corrections * alpha
  end function basic_SGD

end module network_layers