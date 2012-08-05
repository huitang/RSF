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

#ifndef __itkLevelSetEquationSparseRSFTerm_h
#define __itkLevelSetEquationSparseRSFTerm_h

#include "itkLevelSetEquationChanAndVeseGlobalTerm.h"
#include "itkImage.h"

#include "itkNumericTraits.h"

namespace itk
{
/**
 *  \class LevelSetEquationSparseRSFTerm
 *
 *  \tparam TInput Input Image Type
 *  \tparam TLevelSetContainer Level set function container type
 *  \Algorithm from C. Li, C. Kao, J. C. Gore, and Z. Ding. Minimization of region-scalable fitting energy for image
     segmentation. IEEE Trans Image Processing, 17 (10):1940--1949, 2008.
 *  \ingroup ITKLevelSetsv4
 */
template< class TInput, // Input image or mesh
          class TLevelSetContainer >
class LevelSetEquationSparseRSFTerm :
    public LevelSetEquationChanAndVeseGlobalTerm< TInput, TLevelSetContainer >
{
public:
  typedef LevelSetEquationSparseRSFTerm                   Self;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;
  typedef LevelSetEquationChanAndVeseGlobalTerm<
                                    TInput,
                                    TLevelSetContainer >  Superclass;

  /** Method for creation through object factory */
  itkNewMacro( Self );

  /** Run-time type information */
  itkTypeMacro( LevelSetEquationSparseRSFTerm,
                LevelSetEquationChanAndVeseGlobalTerm );

  typedef typename Superclass::InputImageType     InputImageType;
  typedef typename Superclass::InputImagePointer  InputImagePointer;
  typedef typename Superclass::InputPixelType     InputPixelType;
  typedef typename Superclass::InputPixelRealType InputPixelRealType;

  typedef typename Superclass::LevelSetContainerType      LevelSetContainerType;
  typedef typename Superclass::LevelSetContainerPointer   LevelSetContainerPointer;
  typedef typename Superclass::LevelSetType               LevelSetType;
  typedef typename Superclass::LevelSetPointer            LevelSetPointer;
  typedef typename Superclass::LevelSetOutputPixelType    LevelSetOutputPixelType;
  typedef typename Superclass::LevelSetOutputRealType     LevelSetOutputRealType;
  typedef typename Superclass::LevelSetInputIndexType     LevelSetInputIndexType;
  typedef typename Superclass::LevelSetGradientType       LevelSetGradientType;
  typedef typename Superclass::LevelSetHessianType        LevelSetHessianType;
  typedef typename Superclass::LevelSetIdentifierType     LevelSetIdentifierType;

  typedef typename Superclass::HeavisideType              HeavisideType;
  typedef typename Superclass::HeavisideConstPointer      HeavisideConstPointer;

  typedef typename Superclass::LevelSetDataType LevelSetDataType;

  typedef typename Superclass::DomainMapImageFilterType   DomainMapImageFilterType;
  typedef typename Superclass::CacheImageType             CacheImageType;


  typedef typename DomainMapImageFilterType::DomainMapType::const_iterator  DomainIteratorType;

  typedef typename LevelSetContainerType::IdListType          IdListType;
  typedef typename LevelSetContainerType::IdListIterator      IdListIterator;
  typedef typename LevelSetContainerType::IdListConstIterator IdListConstIterator;

  itkSetMacro( GaussianBlurScale, InputPixelRealType );
  itkGetConstMacro( GaussianBlurScale, InputPixelRealType );

  virtual void Update();

  /** Initialize parameters in the terms prior to an iteration */
  virtual void InitializeParameters();

  /** Initialize term parameters in the dense case by computing for each pixel location */
  virtual void Initialize( const LevelSetInputIndexType& iP );

  virtual void UpdatePixel( const LevelSetInputIndexType& iP,
                           const LevelSetOutputRealType & oldValue,
                           const LevelSetOutputRealType & newValue );



protected:
  LevelSetEquationSparseRSFTerm();
  virtual ~LevelSetEquationSparseRSFTerm();

  /** Returns the term contribution for a given location iP, i.e.
   *  \f$ \omega_i( p ) \f$. */
  virtual LevelSetOutputRealType Value( const LevelSetInputIndexType& iP );

  /** Returns the term contribution for a given location iP, i.e.
   *  \f$ \omega_i( p ) \f$. */
  virtual LevelSetOutputRealType Value( const LevelSetInputIndexType& iP,
                                        const LevelSetDataType& iData );
  
  /** Accumulate contribution to term parameters from a given pixel */
  void Accumulate( const InputPixelType& iPix, const LevelSetOutputRealType& iHIn, const LevelSetOutputRealType& iHEx );

  InputPixelRealType CalculateVariance(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData, InputImagePointer bluredMeanImage, InputImagePointer bluredMeanSquareImage);
  void GetCurrentHeavisideImage();
  void GenerateImage(InputImagePointer image);
  void UpdateMeanImage();

  void operator = ( const Self& ); // purposely not implemented
  InputImagePointer    m_BackgroundMeanImage;
  InputImagePointer    m_ForegroundMeanImage;
  InputImagePointer    m_BluredBackgroundSquareMeanImage;
  InputImagePointer    m_BluredForegroundSquareMeanImage;
  InputImagePointer    m_BluredBackgroundMeanImage;
  InputImagePointer    m_BluredForegroundMeanImage;
  InputImagePointer    m_CurrentInverseHeaviside;
  InputImagePointer    m_CurrentHeaviside;
  InputImagePointer    m_CurrentLevelSet;
  InputPixelRealType   m_GaussianBlurScale;


private:
  LevelSetEquationSparseRSFTerm( const Self& ); // purposely not implemented

};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLevelSetEquationSparseRSFTerm.hxx"
#endif

#endif
