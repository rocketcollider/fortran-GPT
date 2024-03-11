
module structured_network
  use network_utils
  use network_layers
  implicit none

  type network
    class(layer), allocatable :: layers(:)
    class(Loss), allocatable :: cost_function
    integer :: outputs
  contains
    procedure :: run => interpret1,interpret2
    procedure :: batch_train => batch_train2
    procedure :: initiate => random_network
    final :: destructor
  end type network

  interface network
    module procedure network_from_layers
    module procedure network_from_layers_and_loss
    module procedure network_from_array_and_layers_and_loss
  end interface network

contains

  subroutine destructor(this)
    type(network) :: this
    print *, 'here network be destructed'
  end subroutine destructor

  subroutine random_network(this)
    class(network), intent(inout) :: this
    integer :: i
    do i=1,size(this%layers)
      call this%layers(i)%random_weights
    end do
  end subroutine random_network

  function network_from_layers(layers) result (out)
    class(layer), intent(in) :: layers(:)
    type(network) :: out
    out%outputs = layers(size(layers))%outputs
    allocate(out%layers, source=layers)
    out%layers = layers

  end function network_from_layers

  function network_from_layers_and_loss(layers, cost) result (out)
    class(layer),intent(in) :: layers(:)
    class(Loss),intent(in) :: cost
    type(network) :: out
    out%outputs = layers(size(layers))%outputs
    allocate(out%cost_function, source=cost)

    allocate(out%layers, mold=layers)
    out%layers = layers
    print *, 'here were started?'
  end function network_from_layers_and_loss

  function network_from_array_and_layers_and_loss(shapes, layers, cost) result (out)
    class(layer), intent(in) :: layers(:)
    class(Loss), intent(in) :: cost
    integer, intent(in) :: shapes(size(layers)+1)
    type(network) :: out
    integer :: i
    allocate(out%cost_function, source=cost)

    allocate(out%layers, mold=layers)
    out%layers = layers
    do i=1,size(layers)
      call out%layers(i)%set_layout(shapes(i), shapes(i+1))
    end do

  end function network_from_array_and_layers_and_loss

  function interpret1(this, signals) result (out)
    class(network), intent(in) :: this
    real, intent(in) :: signals(:)
    real :: out(this%outputs)
    out = reshape(this%interpret2(reshape(signals, [size(signals),1])), [this%outputs])
  end function interpret1

  function interpret2(this, signals) result (out)
    class(network), intent(in) :: this
    real, intent(in) :: signals(:,:)
    !real :: out(this%layers(size(this%layers))%outputs, size(signals,2))
    real, allocatable :: out(:,:)
    integer :: i

    out = signals

    do i=1,size(this%layers)
      out=this%layers(i)%run(out)
    end do
  end function interpret2

  function batch_train2(this, signals, answers, alpha) result (loss)
    class(network), intent(inout) :: this
    real, intent(in) :: signals(:,:), answers(:,:)
    real :: alpha, loss!(this%layers(size(this%layers))%outputs, size(signals,2))
    real, allocatable :: tmp(:,:)
    integer :: i
    print *, 'inside batch train', allocated(tmp), shape(signals)
    allocate(tmp, source=signals)
    !allocate(tmp(size(signals,1),size(signals,2)))

    print *, 'after assign, shapes: ', shape(this%layers), '(should be rank1) ', shape(tmp)
    do i=1,size(this%layers)
      print *, 'before forward train'
      tmp = this%layers(i)%train_forward(tmp)
      print *, 'after forward train'
    end do
    loss = sum(this%cost_function%eval(tmp, answers))
    tmp = this%cost_function%grad(tmp, answers)

    do i=size(this%layers),1,-1
      print *,"before layers train"
      tmp = this%layers(i)%train_backward(tmp, alpha)
      print *, 'after layers train'
      !block
      !  real :: prev_errors(2*this%layers(i)%inputs,2*size(answers,2))
        !tmp = this%layers(i)%train_backward(tmp, alpha)
        !deallocate(tmp)
        !allocate(tmp, source=prev_errors)
      !end block
    end do
    !deallocate(tmp)
  end function batch_train2

end module structured_network