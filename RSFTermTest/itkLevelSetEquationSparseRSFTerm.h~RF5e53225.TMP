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

#include "itkLevelSetEquationChanAndVeseTerm.h"
#include "itkLevelSetEquationTermBase.h"
#include <itkImage.h>

#include "itkNumericTraits.h"


namespace itk
{
/**
 *  \class LevelSetEquationSparseRSFTerm
 *  \brief Class to represent the internal energy Chan And Vese term
 *
 *  \f[
 *    \delta_{\epsilon}\left( \phi_{k} \left( p \right) \right) \cdot
      \left\| I\left( p \right) - \mu_{in} \right\|^2
 *  \cdot
 *  \f]
 *
 *  \li \f$ \delta_{epsilon}  \f$ is a regularized dirac function,
 *  \li \f$ k \f$ is the current level-set id,
 *  \li \f$ I\left( p \right) \f$ is the pixel value at the given location \f$ p \f$,
 *  \li \f$ \mu_{in}  \f$ is the internal mean intensity.
 *
 *  \tparam TInput Input Image Type
 *  \tparam TLevelSetContainer Level set function container type
 *
 *  \ingroup ITKLevelSetsv4
 */
template< class TInput, // Input image or mesh
          class TLevelSetContainer >
class LevelSetEquationSparseRSFTerm :
    public LevelSetEquationChanAndVeseTerm< TInput, TLevelSetContainer >
{
public:
  typedef LevelSetEquationSparseRSFTerm         Self;
  typedef SmartPointer< Self >                            Pointer;
  typedef SmartPointer< const Self >                      ConstPointer;
  typedef LevelSetEquationChanAndVeseTerm< TInput,
                                    TLevelSetContainer >  Superclass;

  /** Method for creation through object factory */
  itkNewMacro( Self );

  /** Run-time type information */
  itkTypeMacro( LevelSetEquationSparseRSFTerm,
                LevelSetEquationChanAndVeseTerm );

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
  itkGetMacro( GaussianBlurScale, InputPixelRealType );
  itkSetMacro( StepForSavingIntermedialResult, InputPixelRealType );
  itkGetMacro( StepForSavingIntermedialResult, InputPixelRealType ); 
  /*itkSetMacro( MeanInternal, InputPixelRealType );
  itkGetMacro( MeanInternal, InputPixelRealType );
  itkSetMacro( MeanExternal, InputPixelRealType );
  itkGetMacro( MeanExternal, InputPixelRealType );
  itkSetMacro( InternalCoefficient, InputPixelRealType );
  itkGetMacro( InternalCoefficient, InputPixelRealType );
  itkSetMacro( ExternalCoefficient, InputPixelRealType );*/
  /** Update the term parameter values at end of iteration */
  virtual void Update();

  /** Initialize parameters in the terms prior to an iteration */
  virtual void InitializeParameters();

  /** Initialize term parameters in the dense case by computing for each pixel location */
  virtual void Initialize( const LevelSetInputIndexType& iP );

  /** Compute the product of Heaviside functions in the multi-levelset cases */
  //virtual void ComputeProductInternal( const LevelSetInputIndexType& iP,
   //                           LevelSetOutputRealType& prod );
 // virtual void ComputeProductExternal( const LevelSetInputIndexType& iP,
	//  LevelSetOutputRealType& prod );
  /** Compute the product of Heaviside functions in the multi-levelset cases
   *  except the current levelset */
 // virtual void ComputeProductTermInternal( const LevelSetInputIndexType& ,
 //                                 LevelSetOutputRealType& )
//  {}
 // virtual void ComputeProductTermExternal( const LevelSetInputIndexType& ,
	//  LevelSetOutputRealType&  );

  /** Supply updates at pixels to keep the term parameters always updated */
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
  //InputPixelType CalculateVarience(const LevelSetInputIndexType& iP, const LevelSetDataType& iData,const InputPixelType& meanValue);

  /** Accumulate contribution to term parameters from a given pixel */
  void Accumulate( const InputPixelType& iPix, const LevelSetOutputRealType& iHIn, const LevelSetOutputRealType& iHEx );
  InputPixelRealType CalculateVarienceFore(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData);
  InputPixelRealType CalculateVarienceBack(const LevelSetInputIndexType& iP, const LevelSetOutputRealType& iData);
  void GetCurrentHeavisideImage();
  void GenerateImage(InputImagePointer image);
  void UpdateMeanImage();


 // InputPixelRealType      m_MeanInternal;
 // InputPixelRealType      m_TotalValueInternal;
 // LevelSetOutputRealType  m_TotalHInternal;
   InputImagePointer     m_BackgroundMeanImage;
   InputImagePointer     m_ForegroundMeanImage;
   InputImagePointer     m_BluredBackgroundSquareMeanImage;
   InputImagePointer     m_BluredForegroundSquareMeanImage;
   InputImagePointer     m_BluredBackgroundMeanImage;
   InputImagePointer     m_BluredForegroundMeanImage;
   InputImagePointer     m_CurrentHeavisideInverse;
   InputImagePointer     m_CurrentHeaviside;
   InputImagePointer     m_CurrentLevelSet;
 // InputPixelRealType      m_MeanExternal;
 // InputPixelRealType      m_TotalValueExternal;
//  LevelSetOutputRealType  m_TotalHExternal;
//  LevelSetOutputRealType   m_InternalCoefficient;
//  LevelSetOutputRealType   m_ExternalCoefficient;
 //  LevelSetConverterPointer  m_LevelSetConverter;
     InputPixelRealType m_GaussianBlurScale;
	 InputPixelRealType m_StepForSavingIntermedialResult;

	 InputPixelRealType   currentIteration;
private:
  LevelSetEquationSparseRSFTerm( const Self& ); // purposely not implemented
  void operator = ( const Self& ); // purposely not implemented
};

}
#ifndef ITK_MANUAL_INSTANTIATION
#include "itkLevelSetEquationSparseRSFTerm.hxx"
#endif

#endif
