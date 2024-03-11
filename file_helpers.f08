module file_helpers
  type :: file
    !private
    integer :: io=-1, buffer_len=80
    character(len=:), allocatable :: dictionary
  contains
    procedure :: readline => read_line
    procedure :: dictionarize => dictionarize
    !final :: destructor !this fires too early!
  end type file

  interface file
    procedure :: open_path
  end interface file

contains

  function open_path(path) result (out)
    character(*), intent(in) :: path
    type(file) :: out
    logical :: exists
    integer :: io
    do io=1,1000
      inquire(unit=io,opened=exists)
      if (.not. exists) then
        open(unit=io, file=path, status='old', action='read')
        out%io = io
        exit
      end if
    end do
  end function open_path

  subroutine destructor(this)
    type(file) :: this
    if (this%io /= -1) then
      close(this%io)
    end if
  end subroutine destructor

  function dictionarize(this) result (intlist)
    class(file),intent(inout) :: this
    character(len=:), allocatable :: dictionary
    integer :: iostat, size_read,i, idx, dic_len, list_pos
    character(len=80) :: buffer
    integer, allocatable :: intlist(:), tmp(:)

    dic_len=0
    allocate(intlist(this%buffer_len))
    intlist(1)=0 ! start with delimiter
    list_pos=2
    if (allocated(this%dictionary)) then
      dictionary = this%dictionary // this%dictionary
      dic_len=len(this%dictionary)
    else
      allocate(character(this%buffer_len):: this%dictionary)
      allocate(character(this%buffer_len*2):: dictionary)
    end if

    rewind(this%io)

    do while (.true.)
      read(this%io, '(A)',iostat=iostat,size=size_read,advance='no') buffer
      if (is_iostat_end(iostat)) then ! eof = 'end of file'
        print *, 'found file end!'
        if (intlist(list_pos-1) == 0) then
          intlist = intlist(:list_pos-1)
        else
          intlist(list_pos) = 0 ! use 0 as delimiter
          intlist = intlist(:list_pos)
        endif
        exit
      endif
      do i=1,size_read
        idx = index(dictionary(:dic_len),buffer(i:i))
        !print *, buffer(i:i), idx, dictionary
        if (idx<1) then
          if (dic_len >= len(dictionary)) then
            dictionary = dictionary // dictionary
          endif
          dictionary = dictionary(1:dic_len)//buffer(i:i)//dictionary(dic_len+2:)

          dic_len=dic_len + 1
          idx=dic_len
        endif

        intlist(list_pos) = idx

        list_pos = list_pos+1
        if (list_pos+1>size(intlist,1)) then
          allocate(tmp, source=intlist)
          deallocate(intlist)
          allocate(intlist(2*size(tmp,1)), source=tmp)
          deallocate(tmp)
        endif

      end do
      if (is_iostat_eor(iostat)) then ! eor = 'end of recorrd'
        intlist(list_pos) = 0 ! use 0 as delimiter
        list_pos = list_pos+1
      else
        exit ! iostat in some other error-state
      end if
    end do
    deallocate(this%dictionary)
      allocate(character(dic_len):: this%dictionary)
    this%dictionary = dictionary(:dic_len)
  end function dictionarize

  function read_line(this) result (line)
    class(file), intent(in) :: this
    character(len=:), allocatable :: line
    integer :: iostat, size_read
    character(len=80) :: buffer
    !character(len=this%buffer_len) :: buffer

    line=''
    buffer=''
    do
      read(this%io, '(A)', &
        iostat = iostat,   &
        !iomsg = iomsg,     &
        size = size_read,  &
        advance = 'no'     &
      ) buffer

      if (iostat == 0) then
        line = line // buffer
      else if (is_iostat_eor(iostat)) then ! eor = 'end of recorrd'
        line = line // buffer(:size_read) ! string-concatenation
        iostat = 0
        exit
      else if (is_iostat_end(iostat)) then ! eof = 'end of file'
        line = line // buffer(:size_read)
        exit
      else
        exit ! iostat in some other error-state
      end if
    end do
  end function read_line
end module file_helpers