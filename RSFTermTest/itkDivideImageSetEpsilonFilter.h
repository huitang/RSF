/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef __itkDivideImageEpsilonFilter_h
#define __itkDivideImageEpsilonFilter_h

#include "itkBinaryFunctorImageFilter.h"
#include "itkNumericTraits.h"

namespace itk
{
namespace Functor
{
/**
 * \class Div
 * \brief
 * \ingroup ITKImageIntensity
 */
template< class TInput1, class TInput2, class TOutput >
class DivEpsilon
{
public:
  DivEpsilon() {}
  ~DivEpsilon() {}
  bool operator!=(const DivEpsilon &) const
  {
    return false;
  }

  bool operator==(const DivEpsilon & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()(const TInput1 & A, const TInput2 & B) const
  {
    typedef typename NumericTraits< TInput1 >::RealType RealType1;
    typedef typename NumericTraits< TInput2 >::RealType RealType2;

    if ( B != NumericTraits< TInput2 >::Zero )
      {
      return static_cast< TOutput >(
        static_cast< RealType1 >( A ) /
        static_cast< RealType2 >( B + 1e-10 ) );
      }
    else
      {
      return NumericTraits< TOutput >::max();
      }
  }
};
}
/** \class DivideImageEpsilonFilter
 * \brief Pixel-wise division of two images.
 *
 * This class is templated over the types of the two
 * input images and the type of the output image. When the divisor is zero,
 * the division result is set to the maximum number that can be
 * represented  by default to avoid exception. Numeric conversions
 * (castings) are done by the C++ defaults.

 */
template< class TInputImage1, class TInputImage2, class TOutputImage >
class ITK_EXPORT DivideImageEpsilonFilter:
  public
  BinaryFunctorImageFilter< TInputImage1, TInputImage2, TOutputImage,
                            Functor::DivEpsilon<
                              typename TInputImage1::PixelType,
                              typename TInputImage2::PixelType,
                              typename TOutputImage::PixelType >   >
{
public:
  /**
   * Standard "Self" typedef.
   */
  typedef DivideImageEpsilonFilter Self;

  /**
   * Standard "Superclass" typedef.
   */
  typedef BinaryFunctorImageFilter< TInputImage1, TInputImage2, TOutputImage,
                                    Functor::DivEpsilon<
                                      typename TInputImage1::PixelType,
                                      typename TInputImage2::PixelType,
                                      typename TOutputImage::PixelType >
                                    > Superclass;

  /**
   * Smart pointer typedef support
   */
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /**
   * Method for creation through the object factory.
   */
  itkNewMacro(Self);
  itkSetMacro( Epsilon,  typename TInputImage1::PixelType );
  itkGetMacro( Epsilon,  typename TInputImage1::PixelType );
  /** Runtime information support. */
  itkTypeMacro(DivideImageEpsilonFilter,
               BinaryFunctorImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro( IntConvertibleToInput2Check,
                   ( Concept::Convertible< int, typename TInputImage2::PixelType > ) );
  itkConceptMacro( Input1Input2OutputDivisionOperatorsCheck,
                   ( Concept::DivisionOperators< typename TInputImage1::PixelType,
                                                 typename TInputImage2::PixelType,
                                                 typename TOutputImage::PixelType > ) );
  /** End concept checking */
#endif
protected:
  DivideImageEpsilonFilter() {this->m_Epsilon=1e-10;}
  virtual ~DivideImageEpsilonFilter() {}

  void GenerateData()
    {
    const typename Superclass::DecoratedInput2ImagePixelType *input
       = dynamic_cast< const typename Superclass::DecoratedInput2ImagePixelType * >(
        this->ProcessObject::GetInput(1) );
    if( input != NULL && input->Get() == itk::NumericTraits< typename TInputImage2::PixelType >::Zero )
      {
      itkGenericExceptionMacro(<<"The constant value used as denominator should not be set to zero");
      }
    else
      {
      Superclass::GenerateData();
      }
    }

private:
  DivideImageEpsilonFilter(const Self &); //purposely not implemented
  void operator=(const Self &); //purposely not implemented
  typename TInputImage1::PixelType  m_Epsilon;

};
} // end namespace itk

#endif
