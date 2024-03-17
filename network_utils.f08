
module network_utils
  implicit none

  type :: wrapper
    class(Activation), pointer :: f
  end type

  interface wrap
    module procedure :: new_wrapper
  end interface wrap

  interface rolling_average
    module procedure :: rolling_average2
    module procedure :: rolling_average3
  end interface rolling_average

  type, abstract :: smooth_function
  end type smooth_function

  type, abstract, extends(smooth_function) :: Activation
  contains
    generic, public  :: activate => activate1, activate2
    procedure :: activate1 => generic_smooth_eval
    procedure(activation_function2), nopass, deferred :: activate2
    procedure(activation_derivative), nopass, deferred :: derivative
  end type Activation

  abstract interface
    pure function activation_function1lalala(signal) result (out)
      real, intent(in), dimension(:) :: signal
      real, dimension(size(signal,1)) :: out
    end function activation_function1lalala

    pure function activation_function2(signal) result (out)
      real, intent(in), dimension(:,:) :: signal
      real, dimension(size(signal,1),size(signal,2)) :: out
    end function activation_function2

    pure function activation_derivative(signal) result (out)
      real, intent(in), dimension(:) :: signal
      real, dimension(size(signal)) :: out
    end function activation_derivative
  end interface

  type, extends(Activation) :: identity
  contains
    procedure, nopass :: activate2 => ident_activate
    procedure, nopass :: derivative => ident_der
  end type identity

  type, extends(Activation) :: ReLU
  contains
    procedure, nopass :: activate2 => a_relu_r !, relu_r
    procedure, nopass :: derivative => a_d_relu_r, d_relu_r
  end type ReLU

  type, extends(Activation) :: softmax
  contains
    procedure, nopass :: activate2 => a_softmax_r
    procedure, nopass :: derivative => a_d_softmax_r
  end type softmax

  type, abstract, extends(smooth_function) :: Loss
  contains
    procedure(loss_function), nopass, deferred :: eval
    procedure(loss_gradient), nopass, deferred :: grad
  end type Loss

  abstract interface
    pure elemental function loss_function(compare, desired) result (out)
      real, intent(in):: compare, desired
      real :: out
    end function loss_function

    pure elemental function loss_gradient(compare, desired) result (out)
      real, intent(in):: compare, desired
      real :: out
    end function loss_gradient
  end interface

  type, extends(Loss) :: SquaredLoss
  contains
    procedure, nopass :: eval => eval_squared_loss
    procedure, nopass :: grad => gradient_squared_loss
  end type SquaredLoss

  type, extends(Loss) :: LogLoss
  contains
    procedure, nopass :: eval => eval_log_loss
    procedure, nopass :: grad => gradient_log_loss
  end type LogLoss

contains

  pure function generic_smooth_eval(this, signal) result (out)
    class(Activation), intent(in) :: this
    real, intent(in) :: signal(:)
    real :: out(size(signal))
    out = reshape(this%activate2(reshape(signal, [size(signal),1])), [size(signal)])
  end function generic_smooth_eval

  pure elemental function eval_squared_loss(compare, desired) result (out)
    real, intent(in) :: compare, desired
    real :: out
    out = ( desired - compare )**2/2.0
  end function eval_squared_loss

  pure elemental function gradient_squared_loss(compare, desired) result (out)
    real, intent(in) :: compare, desired
    real :: out
    out = compare - desired
  end function gradient_squared_loss

  pure elemental function eval_log_loss(compare, desired) result (out)
    real, intent(in) :: compare, desired
    real :: out
    out = - desired * log(compare)
  end function eval_log_loss

  pure elemental function gradient_log_loss(compare, desired) result (out)
    real, intent(in) :: compare, desired
    real :: out
    out = - desired/compare
  end function gradient_log_loss

  function new_wrapper(acti) result (out)
    class(Activation), intent(in), target :: acti
    type(wrapper) :: out
    out%f => acti
  end function new_wrapper

  pure function ident_activate(signal) result (out)
    real, intent(in), dimension (:,:) :: signal
    real, dimension(size(signal,1),size(signal,2)) :: out
    out = signal
  end function ident_activate

  pure function ident_der(signal) result (out)
    real, intent(in), dimension (:) :: signal
    real, dimension(size(signal,1)) :: out
    out = 1.
  end function ident_der

  pure function a_d_relu_r(signal) result (out)
    real, intent(in), dimension (:) :: signal
    real, dimension(size(signal,1)) :: out
    out = merge(1.,0.,signal>0)
  end function a_d_relu_r

  pure function a_relu_r(signal) result (out)
    real, intent(in) :: signal(:,:)
    real, dimension(size(signal,1),size(signal,2)) :: out

    out = merge(signal,0.,signal>0.)
  end function a_relu_r

  elemental real function relu_r(signal)
    real, intent(in) :: signal

    relu_r = max(0.,signal)
  end function relu_r

  elemental function d_relu_r(signal) result (out)
    real, intent(in) :: signal
    real :: out

    out = merge(1.,0., signal > 0. )
  end function d_relu_r


  pure function a_softmax_r(signal) result (out)
    real, intent(in) :: signal(:,:)
    real, dimension(size(signal,1),size(signal,2)) :: out, base
    base = exp(signal)
    out = base / spread(sum(base,1), 1, size(signal,1))
  end function a_softmax_r


  pure function a_d_softmax_r(signal) result (out)
    real, intent(in), dimension(:) :: signal
    real, dimension(size(signal)) :: out
    real :: base
    base = sum(exp(signal))
    out = exp(signal) * (base - exp(signal))/base**2
  end function a_d_softmax_r

  pure function one_hot(state,max) result (out)
    integer, intent(in) :: state, max
    real, dimension(max) :: out
    out = 0.

    !assert state <= max

    if (state > 0 .and. state<=max) then
      out(state) = 1.
    end if
    ! return all 0 automatically
  end function one_hot

  function one_cold(state) result (out)
    real, intent(in) :: state(:)
    integer :: i, out
    out=0

    do i=1,size(state)
      if (state(i) > 0.5) then
        out = i
        exit
      end if
    end do
  end function one_cold

  pure function multi_hot(state,max) result (out)
    integer, intent(in) :: state(:), max
    real :: out(max*size(state))
    integer :: i
    out = 0.

    !assert state <= max
    
    do concurrent (i=1:size(state))
      if (state(i) > 0 .and. state(i)<=max) then
        out(state(i)) = 1.
      end if
      ! return all 0 automatically
    end do
  end function multi_hot

  function get_sort_indices(array) result (out)
    real, intent(inout) :: array(:)
    real, target :: r1(size(array)), r2(size(array))
    integer :: i,n,k, u,v, bin, last, out(size(array))
    integer, target :: i1(size(array)), i2(size(array))
    real, pointer :: unsorted(:), sorted(:)
    integer, pointer :: iunsorted(:), isorted(:)
    last=size(array)

    !just to assing pointers
    sorted => r1
    isorted => i1
    r1=array
    do concurrent (i=1:size(array))
      i1(i) = i
    end do
    do i=0,127 ! increase exponent
      bin = 2**i
      if (bin >= size(array)) then
        exit
      end if

      if (modulo(i,2)<1) then ! if m is even
        unsorted=>r1
        !r2=0
        sorted=>r2
        iunsorted=>i1
        isorted=>i2
        i2=0
      else
        unsorted=>r2
        !r1=0
        sorted=>r1
        iunsorted=>i2
        isorted=>i1
      end if

      ! n = current bin-pair
      do n=0,size(array)-bin-1,2*bin ! run through whole array in steps of 2*binsize
        u = 1
        v = 1
        ! k = next sorted position
        do k = 1,2*bin ! run through pair
          if(unsorted(n+u)>unsorted(n+bin+v)) then

            sorted(n+k) = unsorted(n+bin+v)
            isorted(n+k) = iunsorted(n+bin+v)
            v = v+1

            if (v > bin .or. v+bin+n>last .and. u <= bin) then ! .and. k+1 < 2**(m+1) is _hopefully_ unnecessarry, bc always same as k+1 < 2**(m+1)
              sorted(n+k+1:min(n+2*bin, last)) = unsorted(n+u:n+bin)
              isorted(n+k+1:min(n+2*bin, last)) = iunsorted(n+u:n+bin)
              exit
            end if
          else

            sorted(n+k) = unsorted(n+u)
            isorted(n+k) = iunsorted(n+u)
            u = u+1

            if (u > bin)then! .and. v <= 2**m) then ! basically "==" but defensive.
              sorted(n+k+1:min(n+2*bin, last)) = unsorted(n+bin+v:min(n+2*bin,last))
              isorted(n+k+1:min(n+2*bin, last)) = iunsorted(n+bin+v:min(n+2*bin,last))
              exit
            end if
          end if
        end do
      end do
      if (n <= size(array)) then
        sorted(n+1:) = unsorted(n+1:)
        isorted(n+1:) = iunsorted(n+1:)
      end if

    end do
    array = sorted
    out=isorted
  end function get_sort_indices

  function random_select(distribution) result (out)
    real,intent(in) :: distribution(:)
    real :: score, rrand, sorted(size(distribution))
    integer :: i, out, map(size(distribution))

    out = 0

    sorted = distribution
    map = get_sort_indices(sorted)

    rrand = rand() * sum(distribution)
    score = 0
    do i=1,size(distribution)
      score = score + sorted(i)
      if (score>rrand) then
        out = map(i)
        return
      end if
    end do
  end function random_select

  function rolling_average2(data) result  (out)
    real, intent(in) :: data(:,:)
    real :: out(size(data,1), size(data,2)), tri(size(data,2),size(data,2))
    integer :: i
    tri = 0

    do concurrent (i=1:size(data,2))
      tri(i,1:i) = (1.0/i)
    end do
    out = transpose(matmul(tri,transpose(data)))
  end function rolling_average2

  function rolling_average3(data) result (out)
    real, intent(in) :: data(:,:,:)
    real :: out(size(data,1), size(data,2), size(data,3)), tri(size(data,2), size(data,2))
    integer :: i
    tri = 0

    do concurrent (i=1:size(data,2))
      tri(i,1:i) = (1.0/i)
    end do

    do concurrent (i=1:size(data,3))
      out(:,:,i) = transpose(matmul(tri,transpose(data(:,:,i))))
    end do
  end function rolling_average3

end module network_utils