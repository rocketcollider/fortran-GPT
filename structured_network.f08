
module structured_network
  use network_utils
  use network_layers
  implicit none

  type network
    class(layer), allocatable :: layers(:)
    class(Loss), allocatable :: cost_function
    integer :: outputs
  contains
    generic :: run => interpret2, interpret1
    procedure :: interpret2 => interpret2
    procedure :: interpret1 => interpret1
    procedure :: batch_train => batch_train2
    procedure :: initiate => random_network
  end type network

  interface network
    module procedure network_from_layers
    module procedure network_from_layers_and_loss
    module procedure network_from_array_and_layers_and_loss
  end interface network

contains

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

  end function network_from_layers

  function network_from_layers_and_loss(layers, cost) result (out)
    class(layer),intent(in) :: layers(:)
    class(Loss),intent(in) :: cost
    type(network) :: out
    out%outputs = layers(size(layers))%outputs
    allocate(out%cost_function, source=cost)

    allocate(out%layers, source=layers)
  end function network_from_layers_and_loss

  function network_from_array_and_layers_and_loss(shapes, layers, cost) result (out)
    class(layer), intent(in) :: layers(:)
    class(Loss), intent(in) :: cost
    integer, intent(in) :: shapes(size(layers)+1)
    type(network) :: out
    integer :: i
    allocate(out%cost_function, source=cost)
    out%outputs = shapes(size(shapes))

    allocate(out%layers, source=layers)
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

    allocate(out, source=signals)

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
    allocate(tmp, source=signals)

    do i=1,size(this%layers)
      block
        real :: carrier(this%layers(i)%outputs, size(answers,2))
        carrier = this%layers(i)%train_forward(tmp)
        deallocate(tmp)
        allocate(tmp, source=carrier)
      end block
    end do
    loss = sum(this%cost_function%eval(tmp, answers))/size(signals,2)
    block
      real :: inversion(size(tmp,1), size(tmp,2))
      inversion = this%cost_function%grad(tmp, answers)
      deallocate(tmp)
      allocate(tmp, source=inversion)
    end block

    do i=size(this%layers),1,-1
      block
        real :: prev_error(this%layers(i)%inputs, size(answers,2))
        prev_error = this%layers(i)%train_backward(tmp, alpha)
        deallocate(tmp)
        allocate(tmp, source=prev_error)
      end block
    end do
    deallocate(tmp)
  end function batch_train2

end module structured_network